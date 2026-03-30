import base64
import io
import os
import secrets
from uuid import uuid4

import numpy as np
import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, url_for

from swv_core import (
    FIXED_DERIV_SMOOTH_WIN,
    FIXED_SMOOTH_SIGMA,
    build_output_dataframe,
    figure_to_png_bytes,
    format_polynomial,
    make_plot_corrected,
    make_plot_raw,
    make_plot_step2,
    make_plot_zero_line,
    swv_workflow,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.secret_key = os.environ.get("SWV_SECRET_KEY", secrets.token_hex(16))

UPLOAD_CACHE = {}
OUTPUT_CACHE = {}


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SWV Peak Baseline Correction</title>
  <style>
    :root {
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --line: #d8d0c2;
      --ink: #1f2a30;
      --muted: #56646b;
      --accent: #0e6b62;
      --accent-2: #d97a2b;
      --danger: #9f2d2d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #efe6d6 0, transparent 28%),
        linear-gradient(180deg, #f2eee6 0%, var(--bg) 100%);
    }
    .shell {
      width: min(1220px, calc(100% - 32px));
      margin: 24px auto 48px;
    }
    .hero {
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(14,107,98,0.10), rgba(217,122,43,0.06)), var(--panel);
      box-shadow: 0 12px 30px rgba(31,42,48,0.08);
    }
    h1, h2, h3 { margin: 0 0 12px; }
    p { margin: 0 0 12px; line-height: 1.5; }
    .grid {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 20px;
      margin-top: 20px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 18px;
      box-shadow: 0 10px 24px rgba(31,42,48,0.05);
    }
    .stack > * + * { margin-top: 14px; }
    label {
      display: block;
      font-size: 0.92rem;
      font-weight: 600;
      margin-bottom: 6px;
    }
    input[type="file"], input[type="number"], select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      color: var(--ink);
    }
    input[type="checkbox"] { margin-right: 8px; }
    .button {
      display: inline-block;
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 12px 14px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      text-align: center;
      text-decoration: none;
    }
    .button.secondary { background: var(--accent-2); }
    .hint, .muted { color: var(--muted); font-size: 0.92rem; }
    .error {
      border: 1px solid rgba(159,45,45,0.2);
      color: var(--danger);
      background: rgba(159,45,45,0.06);
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 14px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: #fff;
    }
    .metric .k { font-size: 0.82rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .metric .v { margin-top: 6px; font-size: 1.05rem; font-weight: 700; }
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }
    .selector-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fff;
      padding: 12px;
    }
    .selector-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
      align-items: center;
    }
    .selector-toolbar button {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 8px 12px;
      cursor: pointer;
      color: var(--ink);
    }
    .selector-toolbar button.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .selector-wrap {
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: linear-gradient(180deg, #fff 0%, #faf6ee 100%);
    }
    .selector-wrap svg {
      display: block;
      width: 100%;
      height: auto;
      cursor: crosshair;
    }
    .selector-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .plot-card img {
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid #e8e1d5;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th { background: #f6f1e7; position: sticky; top: 0; }
    .table-wrap { overflow: auto; max-height: 360px; border: 1px solid var(--line); border-radius: 14px; }
    .two-col {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }
    code {
      background: #f0ece4;
      padding: 2px 6px;
      border-radius: 6px;
    }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>SWV Peak Baseline Correction Tool</h1>
      <p>Self-contained local web app for the existing SWV workflow. Upload CSV or Excel, choose the potential/current columns, choose the peak orientation, define the desired peak bounds from the raw data, and export the corrected dataset without Streamlit.</p>
      <p class="hint">Fixed smoothing sigma: <strong>{{ fixed_sigma }}</strong> | Fixed detrended smoothing window: <strong>{{ fixed_deriv_win }}</strong></p>
    </section>

    <div class="grid">
      <section class="panel">
        {% if error %}
          <div class="error">{{ error }}</div>
        {% endif %}
        <form method="post" enctype="multipart/form-data" class="stack">
          <input type="hidden" name="upload_id" value="{{ upload_id or '' }}">
          <div>
            <label for="data_file">Upload CSV or Excel</label>
            <input id="data_file" type="file" name="data_file" accept=".csv,.xlsx,.xls">
          </div>
          {% if numeric_cols %}
            <div>
              <label for="peak_orientation">Peak orientation</label>
              <select id="peak_orientation" name="peak_orientation">
                <option value="downward" {% if form.peak_orientation == "downward" %}selected{% endif %}>Concaving up / pointing downward</option>
                <option value="upward" {% if form.peak_orientation == "upward" %}selected{% endif %}>Concaving down / pointing upward</option>
              </select>
            </div>
            <div>
              <label for="x_col">Potential column</label>
              <select id="x_col" name="x_col">
                {% for col in numeric_cols %}
                  <option value="{{ col }}" {% if form.x_col == col %}selected{% endif %}>{{ col }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <label for="y_col">Current column</label>
              <select id="y_col" name="y_col">
                {% for col in numeric_cols %}
                  <option value="{{ col }}" {% if form.y_col == col %}selected{% endif %}>{{ col }}</option>
                {% endfor %}
              </select>
            </div>
          {% endif %}
          <div class="two-col">
            <div>
              <label for="peak_left_bound">Peak left bound (Potential)</label>
              <input id="peak_left_bound" type="number" step="any" name="peak_left_bound" value="{{ form.peak_left_bound }}">
            </div>
            <div>
              <label for="peak_right_bound">Peak right bound (Potential)</label>
              <input id="peak_right_bound" type="number" step="any" name="peak_right_bound" value="{{ form.peak_right_bound }}">
            </div>
          </div>
          <div class="two-col">
            <div>
              <label for="poly_degree">Zero-line polynomial degree</label>
              <select id="poly_degree" name="poly_degree">
                {% for degree in [1, 2, 3] %}
                  <option value="{{ degree }}" {% if form.poly_degree|int == degree %}selected{% endif %}>{{ degree }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <label for="manual_eps">Manual safety margin ε</label>
              <input id="manual_eps" type="number" min="0" step="0.0000001" name="manual_eps" value="{{ form.manual_eps }}">
            </div>
          </div>
          <div>
            <label><input type="checkbox" name="auto_eps" value="1" {% if form.auto_eps %}checked{% endif %}>Use automatic ε</label>
            <label><input type="checkbox" name="sort_x" value="1" {% if form.sort_x %}checked{% endif %}>Sort by potential</label>
            <label><input type="checkbox" name="drop_na" value="1" {% if form.drop_na %}checked{% endif %}>Drop NaN rows</label>
          </div>
          <button class="button secondary" type="submit" name="action" value="load">Load Data</button>
          {% if selector_data %}
            <button class="button" type="submit" name="action" value="analyze">Run Analysis</button>
          {% endif %}
        </form>
        <p class="hint" style="margin-top:14px;">Load a file first. After that, you can click on the raw trace to set the left and right peak bounds, then run the analysis.</p>
      </section>

      <main class="stack">
        {% if preview_html %}
          <section class="panel">
            <h2>Peak Window Selection</h2>
            <p class="muted">Choose whether your next click sets the left or right bound, then click directly on the raw trace. The numeric inputs update automatically and you can still edit them manually.</p>
            <div class="selector-card">
              <div class="selector-toolbar">
                <button type="button" id="pick-left" class="active">Next click sets left bound</button>
                <button type="button" id="pick-right">Next click sets right bound</button>
                <button type="button" id="clear-bounds">Clear bounds</button>
              </div>
              <div class="selector-wrap">
                <svg id="raw-selector" viewBox="0 0 900 320" preserveAspectRatio="none" aria-label="Raw data selector"></svg>
              </div>
              <div class="selector-meta" id="selector-meta">Click on the chart to place the current bound.</div>
            </div>
          </section>

          <section class="panel">
            <h2>Preview</h2>
            <div class="table-wrap">{{ preview_html|safe }}</div>
          </section>
        {% endif %}

        {% if result %}
          <section class="panel">
            <h2>Detected Peak Summary</h2>
            <div class="metrics">
              <div class="metric"><div class="k">Left Boundary</div><div class="v">{{ result.left_boundary }}</div></div>
              <div class="metric"><div class="k">Apex</div><div class="v">{{ result.apex }}</div></div>
              <div class="metric"><div class="k">Right Boundary</div><div class="v">{{ result.right_boundary }}</div></div>
              <div class="metric"><div class="k">Requested Window</div><div class="v">{{ result.requested_window }}</div></div>
              <div class="metric"><div class="k">Peak Orientation</div><div class="v">{{ result.peak_orientation }}</div></div>
              <div class="metric"><div class="k">Min Gap</div><div class="v">{{ result.min_gap }}</div></div>
              <div class="metric"><div class="k">Safety Margin ε</div><div class="v">{{ result.eps }}</div></div>
              <div class="metric"><div class="k">No Cross</div><div class="v">{{ result.no_cross }}</div></div>
              <div class="metric"><div class="k">No Touch</div><div class="v">{{ result.no_touch }}</div></div>
            </div>
          </section>

          <section class="panel">
            <h2>Step-by-Step Visualization</h2>
            <div class="plot-grid">
              {% for plot in plots %}
                <article class="plot-card">
                  <h3>{{ plot.title }}</h3>
                  <img src="data:image/png;base64,{{ plot.image }}" alt="{{ plot.title }}">
                  {% if plot.note %}
                    <p class="muted">{{ plot.note }}</p>
                  {% endif %}
                </article>
              {% endfor %}
            </div>
          </section>

          <section class="panel">
            <h2>Fit Details</h2>
            <p><strong>Rough apex-detection reference line:</strong> <code>y = {{ result.rough_line }}</code></p>
            <p><strong>Max orthogonal distance:</strong> <code>{{ result.apex_distance }}</code></p>
            <p><strong>Polynomial degree:</strong> <code>{{ result.poly_degree }}</code></p>
            <p><strong>Polynomial coefficients:</strong> <code>{{ result.coefficients }}</code></p>
            <p><strong>Polynomial expression:</strong> <code>{{ result.polynomial_expression }}</code></p>
            <p><strong>Vertical shift applied:</strong> <code>{{ result.delta_shift }}</code></p>
          </section>

          <section class="panel">
            <h2>Processed Data Export</h2>
            <p><a class="button secondary" href="{{ download_url }}">Download corrected CSV</a></p>
            <div class="table-wrap">{{ output_html|safe }}</div>
          </section>
        {% else %}
          <section class="panel">
            <h2>Workflow</h2>
            <p>1. Smooth the SWV trace with fixed sigma = 1.</p>
            <p>2. Use the user-supplied left and right bounds to define the target peak window on the raw data.</p>
            <p>3. Fit a rough upper or lower reference line inside that window based on the selected peak orientation.</p>
            <p>4. Pick the apex by maximum orthogonal distance away from that reference line.</p>
            <p>5. Fit a polynomial zero line using points outside the selected peak window.</p>
            <p>6. Shift the polynomial toward the non-peak side to satisfy no-cross and no-touch.</p>
            <p>8. Export the zero-line-relative corrected curve.</p>
          </section>
        {% endif %}
      </main>
    </div>
  </div>
</body>
{% if selector_data %}
<script>
(() => {
  const selectorData = {{ selector_data | tojson }};
  const svg = document.getElementById("raw-selector");
  const leftInput = document.getElementById("peak_left_bound");
  const rightInput = document.getElementById("peak_right_bound");
  const pickLeftButton = document.getElementById("pick-left");
  const pickRightButton = document.getElementById("pick-right");
  const clearButton = document.getElementById("clear-bounds");
  const meta = document.getElementById("selector-meta");
  if (!svg || !leftInput || !rightInput) return;

  const width = 900;
  const height = 320;
  const padding = { left: 56, right: 24, top: 18, bottom: 34 };
  const potentials = selectorData.potentials;
  const currents = selectorData.currents_uA;
  const xMin = selectorData.x_min;
  const xMax = selectorData.x_max;
  const yMin = selectorData.y_min;
  const yMax = selectorData.y_max;
  let target = "left";

  function setTarget(nextTarget) {
    target = nextTarget;
    pickLeftButton.classList.toggle("active", target === "left");
    pickRightButton.classList.toggle("active", target === "right");
    meta.textContent = target === "left"
      ? "Next click will update the left bound."
      : "Next click will update the right bound.";
  }

  function xScale(value) {
    return padding.left + ((value - xMin) / (xMax - xMin || 1)) * (width - padding.left - padding.right);
  }

  function yScale(value) {
    return height - padding.bottom - ((value - yMin) / (yMax - yMin || 1)) * (height - padding.top - padding.bottom);
  }

  function nearestPotential(clientX) {
    const rect = svg.getBoundingClientRect();
    const ratio = (clientX - rect.left) / rect.width;
    const dataX = xMin + Math.max(0, Math.min(1, ratio)) * (xMax - xMin);
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < potentials.length; i += 1) {
      const dist = Math.abs(potentials[i] - dataX);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return Number(potentials[bestIdx].toFixed(5));
  }

  function render() {
    svg.innerHTML = "";

    const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("x", "0");
    bg.setAttribute("y", "0");
    bg.setAttribute("width", String(width));
    bg.setAttribute("height", String(height));
    bg.setAttribute("fill", "#fffdf8");
    svg.appendChild(bg);

    const axisColor = "#7b877f";
    const lineColor = "#0e6b62";
    const boundColor = "#d97a2b";

    const xAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
    xAxis.setAttribute("x1", String(padding.left));
    xAxis.setAttribute("x2", String(width - padding.right));
    xAxis.setAttribute("y1", String(height - padding.bottom));
    xAxis.setAttribute("y2", String(height - padding.bottom));
    xAxis.setAttribute("stroke", axisColor);
    svg.appendChild(xAxis);

    const yAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
    yAxis.setAttribute("x1", String(padding.left));
    yAxis.setAttribute("x2", String(padding.left));
    yAxis.setAttribute("y1", String(padding.top));
    yAxis.setAttribute("y2", String(height - padding.bottom));
    yAxis.setAttribute("stroke", axisColor);
    svg.appendChild(yAxis);

    const points = potentials.map((x, idx) => `${xScale(x)},${yScale(currents[idx])}`).join(" ");
    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", lineColor);
    polyline.setAttribute("stroke-width", "2");
    polyline.setAttribute("points", points);
    svg.appendChild(polyline);

    const leftValue = leftInput.value === "" ? null : Number(leftInput.value);
    const rightValue = rightInput.value === "" ? null : Number(rightInput.value);
    if (leftValue !== null && rightValue !== null && leftValue < rightValue) {
      const shade = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      shade.setAttribute("x", String(xScale(leftValue)));
      shade.setAttribute("y", String(padding.top));
      shade.setAttribute("width", String(xScale(rightValue) - xScale(leftValue)));
      shade.setAttribute("height", String(height - padding.top - padding.bottom));
      shade.setAttribute("fill", "rgba(217,122,43,0.15)");
      svg.appendChild(shade);
    }

    [leftValue, rightValue].forEach((value, idx) => {
      if (value === null || Number.isNaN(value)) return;
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "line");
      marker.setAttribute("x1", String(xScale(value)));
      marker.setAttribute("x2", String(xScale(value)));
      marker.setAttribute("y1", String(padding.top));
      marker.setAttribute("y2", String(height - padding.bottom));
      marker.setAttribute("stroke", boundColor);
      marker.setAttribute("stroke-width", "2");
      marker.setAttribute("stroke-dasharray", idx === 0 ? "6 4" : "2 3");
      svg.appendChild(marker);
    });
  }

  svg.addEventListener("click", (event) => {
    const value = nearestPotential(event.clientX);
    if (target === "left") {
      leftInput.value = value.toFixed(5);
      setTarget("right");
    } else {
      rightInput.value = value.toFixed(5);
      setTarget("left");
    }
    render();
  });

  pickLeftButton.addEventListener("click", () => setTarget("left"));
  pickRightButton.addEventListener("click", () => setTarget("right"));
  clearButton.addEventListener("click", () => {
    leftInput.value = "";
    rightInput.value = "";
    render();
    setTarget("left");
  });
  leftInput.addEventListener("input", render);
  rightInput.addEventListener("input", render);

  render();
})();
</script>
{% endif %}
</html>
"""


def _default_form_state():
    return {
        "x_col": "",
        "y_col": "",
        "peak_orientation": "downward",
        "peak_left_bound": "",
        "peak_right_bound": "",
        "poly_degree": "2",
        "auto_eps": True,
        "manual_eps": "1e-6",
        "sort_x": True,
        "drop_na": True,
    }


def _bool_field(name):
    return request.form.get(name) == "1"


def _dataframe_preview_html(df):
    return df.head(20).to_html(index=False, classes="data-table", border=0, justify="left")


def _encode_plot(fig):
    return base64.b64encode(figure_to_png_bytes(fig)).decode("ascii")


def _load_dataframe_from_upload(file_storage):
    filename = file_storage.filename or ""
    if not filename:
        raise ValueError("Choose a CSV or Excel file.")

    suffix = os.path.splitext(filename.lower())[1]
    content = file_storage.read()
    if not content:
        raise ValueError("The uploaded file is empty.")

    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(content), skipinitialspace=True)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")

    return filename, df


def _get_numeric_columns(df):
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _build_selector_data(E, I):
    currents_uA = I * 1e6
    y_min = float(np.min(currents_uA))
    y_max = float(np.max(currents_uA))
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    return {
        "potentials": [float(x) for x in E],
        "currents_uA": [float(y) for y in currents_uA],
        "x_min": float(np.min(E)),
        "x_max": float(np.max(E)),
        "y_min": y_min,
        "y_max": y_max,
    }


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}, 200


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    form = _default_form_state()
    upload_id = ""
    df = None
    numeric_cols = []
    preview_html = None
    result_view = None
    plots = []
    output_html = None
    download_url = None
    selector_data = None

    if request.method == "POST":
        form.update(
            {
                "x_col": request.form.get("x_col", ""),
                "y_col": request.form.get("y_col", ""),
                "peak_orientation": request.form.get("peak_orientation", form["peak_orientation"]),
                "peak_left_bound": request.form.get("peak_left_bound", form["peak_left_bound"]),
                "peak_right_bound": request.form.get("peak_right_bound", form["peak_right_bound"]),
                "poly_degree": request.form.get("poly_degree", form["poly_degree"]),
                "manual_eps": request.form.get("manual_eps", form["manual_eps"]),
                "auto_eps": _bool_field("auto_eps"),
                "sort_x": _bool_field("sort_x"),
                "drop_na": _bool_field("drop_na"),
            }
        )

        upload_id = request.form.get("upload_id", "")
        action = request.form.get("action", "load")
        try:
            if "data_file" in request.files and request.files["data_file"].filename:
                filename, df = _load_dataframe_from_upload(request.files["data_file"])
                upload_id = uuid4().hex
                UPLOAD_CACHE[upload_id] = {"filename": filename, "df": df}
            elif upload_id in UPLOAD_CACHE:
                df = UPLOAD_CACHE[upload_id]["df"]
            else:
                raise ValueError("Upload a file to begin.")

            numeric_cols = _get_numeric_columns(df)
            if len(numeric_cols) < 2:
                raise ValueError("Need at least two numeric columns for potential and current.")

            if form["x_col"] not in numeric_cols:
                form["x_col"] = numeric_cols[0]
            if form["y_col"] not in numeric_cols:
                form["y_col"] = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            preview_html = _dataframe_preview_html(df)

            work = df[[form["x_col"], form["y_col"]]].copy()
            work.columns = ["Potential", "Current"]

            if form["drop_na"]:
                work = work.dropna()
            if form["sort_x"]:
                work = work.sort_values("Potential")

            E = work["Potential"].to_numpy(dtype=float)
            I = work["Current"].to_numpy(dtype=float)
            if len(E) < 10:
                raise ValueError("Too few valid points after filtering.")

            selector_data = _build_selector_data(E, I)

            if action == "analyze":
                if form["peak_left_bound"] == "" or form["peak_right_bound"] == "":
                    raise ValueError("Enter both peak left bound and peak right bound, or click them on the raw trace.")

                peak_left_bound = float(form["peak_left_bound"])
                peak_right_bound = float(form["peak_right_bound"])
                poly_degree = int(form["poly_degree"])
                eps = None if form["auto_eps"] else float(form["manual_eps"])

                result = swv_workflow(
                    E,
                    I,
                    peak_left_bound=peak_left_bound,
                    peak_right_bound=peak_right_bound,
                    peak_orientation=form["peak_orientation"],
                    eps=eps,
                    poly_degree=poly_degree,
                )

                out_df = build_output_dataframe(result)
                download_id = uuid4().hex
                OUTPUT_CACHE[download_id] = out_df.to_csv(index=False).encode("utf-8")
                download_url = url_for("download_output", download_id=download_id)
                output_html = _dataframe_preview_html(out_df)

                plots = [
                    {"title": "1. Raw Data", "image": _encode_plot(make_plot_raw(E, I)), "note": None},
                    {
                        "title": "2. Peak Detection",
                        "image": _encode_plot(
                            make_plot_step2(
                                E,
                                I,
                                result["I_smooth"],
                                result["ref_mask"],
                                result["left_idx"],
                                result["apex_idx"],
                                result["right_idx"],
                                result["search_start"],
                                result["search_end"],
                                result["rough_reference_line"],
                            )
                        ),
                        "note": "Points outside the selected peak window are used as the reference points for the final zero-line fit.",
                    },
                    {
                        "title": "3. Zero Line",
                        "image": _encode_plot(
                            make_plot_zero_line(
                                E,
                                I,
                                result["zero_line_fit_curve"],
                                result["zero_line"],
                                result["ref_mask"],
                                result["left_idx"],
                                result["apex_idx"],
                                result["right_idx"],
                            )
                        ),
                        "note": None,
                    },
                    {
                        "title": "4. Relative Curve",
                        "image": _encode_plot(make_plot_corrected(E, result["corrected"], result["left_idx"], result["right_idx"])),
                        "note": None,
                    },
                ]

                result_view = {
                    "left_boundary": f"{E[result['left_idx']]:.5f}",
                    "apex": f"{E[result['apex_idx']]:.5f}",
                    "right_boundary": f"{E[result['right_idx']]:.5f}",
                    "requested_window": f"{result['requested_left_bound']:.5f} to {result['requested_right_bound']:.5f}",
                    "peak_orientation": "Concaving up / pointing downward" if result["peak_orientation"] == "downward" else "Concaving down / pointing upward",
                    "min_gap": f"{result['min_gap']:.5e}",
                    "eps": f"{result['eps']:.5e}",
                    "no_cross": str(bool(result["no_cross"])),
                    "no_touch": str(bool(result["no_touch"])),
                    "rough_line": f"{result['rough_reference_m']:.6e} * E + {result['rough_reference_b']:.6e}",
                    "apex_distance": f"{result['apex_max_distance']:.6e}",
                    "poly_degree": str(result["zero_line_poly_degree"]),
                    "coefficients": np.array2string(result["zero_line_coeffs"], precision=6, separator=", "),
                    "polynomial_expression": format_polynomial(result["zero_line_coeffs"], precision=4),
                    "delta_shift": f"{result['zero_line_delta_shift']:.6e}",
                }
        except Exception as exc:
            error = str(exc)
            if upload_id in UPLOAD_CACHE:
                df = UPLOAD_CACHE[upload_id]["df"]
                numeric_cols = _get_numeric_columns(df)
                preview_html = _dataframe_preview_html(df)

    return render_template_string(
        TEMPLATE,
        error=error,
        form=form,
        fixed_sigma=FIXED_SMOOTH_SIGMA,
        fixed_deriv_win=FIXED_DERIV_SMOOTH_WIN,
        upload_id=upload_id,
        numeric_cols=numeric_cols,
        preview_html=preview_html,
        selector_data=selector_data,
        result=result_view,
        plots=plots,
        output_html=output_html,
        download_url=download_url,
    )


@app.route("/download/<download_id>")
def download_output(download_id):
    csv_bytes = OUTPUT_CACHE.get(download_id)
    if csv_bytes is None:
        return redirect(url_for("index"))
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=swv_corrected_output.csv"},
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
