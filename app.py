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
    swv_downward_workflow,
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
  <title>SWV Downward-Peak Baseline Correction</title>
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
      <h1>SWV Downward-Peak Baseline Correction Tool</h1>
      <p>Self-contained local web app for the existing SWV workflow. Upload CSV or Excel, choose the potential/current columns, tune the peak-search parameters, and export the corrected dataset without Streamlit.</p>
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
              <label for="fallback_frac">Boundary amplitude-drop fraction</label>
              <input id="fallback_frac" type="number" min="0.01" max="0.30" step="0.01" name="fallback_frac" value="{{ form.fallback_frac }}">
            </div>
            <div>
              <label for="edge_exclusion_frac">Edge exclusion fraction</label>
              <input id="edge_exclusion_frac" type="number" min="0.00" max="0.25" step="0.01" name="edge_exclusion_frac" value="{{ form.edge_exclusion_frac }}">
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
          <button class="button" type="submit">Run Analysis</button>
        </form>
        <p class="hint" style="margin-top:14px;">If a file is already loaded, you can change parameters and rerun without uploading again.</p>
      </section>

      <main class="stack">
        {% if preview_html %}
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
            <p><strong>Rough apex-detection line:</strong> <code>y = {{ result.rough_line }}</code></p>
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
            <p>2. Exclude edge regions from apex search.</p>
            <p>3. Fit a rough upper line in the middle region.</p>
            <p>4. Pick the apex by maximum orthogonal distance below that line.</p>
            <p>5. Detect left and right boundaries from detrended curvature plus amplitude drop.</p>
            <p>6. Fit a polynomial zero line using only the local outer points.</p>
            <p>7. Shift the polynomial upward to satisfy no-cross and no-touch.</p>
            <p>8. Export the zero-line-relative corrected curve.</p>
          </section>
        {% endif %}
      </main>
    </div>
  </div>
</body>
</html>
"""


def _default_form_state():
    return {
        "x_col": "",
        "y_col": "",
        "fallback_frac": "0.06",
        "edge_exclusion_frac": "0.10",
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

    if request.method == "POST":
        form.update(
            {
                "x_col": request.form.get("x_col", ""),
                "y_col": request.form.get("y_col", ""),
                "fallback_frac": request.form.get("fallback_frac", form["fallback_frac"]),
                "edge_exclusion_frac": request.form.get("edge_exclusion_frac", form["edge_exclusion_frac"]),
                "poly_degree": request.form.get("poly_degree", form["poly_degree"]),
                "manual_eps": request.form.get("manual_eps", form["manual_eps"]),
                "auto_eps": _bool_field("auto_eps"),
                "sort_x": _bool_field("sort_x"),
                "drop_na": _bool_field("drop_na"),
            }
        )

        upload_id = request.form.get("upload_id", "")
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

            fallback_frac = float(form["fallback_frac"])
            edge_exclusion_frac = float(form["edge_exclusion_frac"])
            poly_degree = int(form["poly_degree"])
            eps = None if form["auto_eps"] else float(form["manual_eps"])

            result = swv_downward_workflow(
                E,
                I,
                fallback_frac=fallback_frac,
                edge_exclusion_frac=edge_exclusion_frac,
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
                            result["rough_upper_line"],
                        )
                    ),
                    "note": "Highlighted outer points are the reference points used for the final zero-line fit.",
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
                "min_gap": f"{result['min_gap']:.5e}",
                "eps": f"{result['eps']:.5e}",
                "no_cross": str(bool(result["no_cross"])),
                "no_touch": str(bool(result["no_touch"])),
                "rough_line": f"{result['rough_upper_m']:.6e} * E + {result['rough_upper_b']:.6e}",
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
        headers={"Content-Disposition": "attachment; filename=swv_downward_corrected_output.csv"},
    )


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
