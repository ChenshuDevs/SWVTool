import io
import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIXED_SMOOTH_SIGMA = 1
FIXED_DERIV_SMOOTH_WIN = 9


def gaussian_smooth(y, sigma_pts=1):
    radius = int(max(3, sigma_pts * 4))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_pts**2))
    kernel /= kernel.sum()
    ypad = np.pad(y, (radius, radius), mode="edge")
    return np.convolve(ypad, kernel, mode="same")[radius:-radius]


def moving_average(y, win=9):
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    ypad = np.pad(y, (win // 2, win // 2), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def second_derivative_nonuniform(x, y):
    return np.gradient(np.gradient(y, x), x)


def format_polynomial(coeffs, precision=4):
    coeffs = np.asarray(coeffs)
    degree = len(coeffs) - 1
    terms = []
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if abs(coeff) < 1e-15:
            continue
        coeff_str = f"{coeff:.{precision}e}"
        if power == 0:
            terms.append(coeff_str)
        elif power == 1:
            terms.append(f"{coeff_str}*E")
        else:
            terms.append(f"{coeff_str}*E^{power}")
    return " + ".join(terms) if terms else "0"


def _validate_peak_orientation(peak_orientation):
    if peak_orientation not in {"downward", "upward"}:
        raise ValueError("peak_orientation must be either 'downward' or 'upward'.")


def fit_envelope_line(E, I, mode="upper", ref_mask=None, eps=None, n_slope=3501):
    E = np.asarray(E)
    I = np.asarray(I)

    if ref_mask is None:
        ref_mask = np.ones_like(I, dtype=bool)
    if eps is None:
        eps = 0.0

    x0, x1 = E[0], E[-1]
    y0, y1 = I[0], I[-1]
    center_slope = (y1 - y0) / (x1 - x0 + 1e-12)
    yspan = np.max(I) - np.min(I)
    xrange = x1 - x0 + 1e-12
    slope_guard = 4 * yspan / xrange

    m_grid = np.linspace(center_slope - slope_guard, center_slope + slope_guard, n_slope)
    E_ref = E[ref_mask]

    best = None
    for m in m_grid:
        if mode == "upper":
            b = np.max(I + eps - m * E)
            score = np.mean(m * E_ref + b)
            is_better = best is None or score < best[0]
        elif mode == "lower":
            b = np.min(I - eps - m * E)
            score = np.mean(m * E_ref + b)
            is_better = best is None or score > best[0]
        else:
            raise ValueError("mode must be either 'upper' or 'lower'.")

        if is_better:
            best = (score, m, b)

    _, m_best, b_best = best
    baseline = m_best * E + b_best
    return {"m": m_best, "b": b_best, "baseline": baseline}


def fit_rough_reference_line_for_apex(E_region, I_region, peak_orientation, n_slope=1201):
    _validate_peak_orientation(peak_orientation)
    mode = "upper" if peak_orientation == "downward" else "lower"
    return fit_envelope_line(
        E_region,
        I_region,
        mode=mode,
        ref_mask=np.ones_like(I_region, dtype=bool),
        eps=0.0,
        n_slope=n_slope,
    )


def choose_apex_by_orthogonal_distance(E_region, I_region, rough_line, peak_orientation):
    _validate_peak_orientation(peak_orientation)
    m = rough_line["m"]
    b = rough_line["b"]
    denom = np.sqrt(m * m + 1.0)

    if peak_orientation == "downward":
        distance = (m * E_region - I_region + b) / denom
    else:
        distance = (I_region - m * E_region - b) / denom

    local_idx = int(np.argmax(distance))
    return {
        "local_idx": local_idx,
        "distance": distance,
        "max_distance": float(distance[local_idx]),
    }


def _find_closest_index(E, target):
    return int(np.argmin(np.abs(E - target)))


def find_peak_in_bounds(E, I, left_bound, right_bound, peak_orientation="downward"):
    _validate_peak_orientation(peak_orientation)
    Is = gaussian_smooth(I, sigma_pts=FIXED_SMOOTH_SIGMA)
    left_bound = float(left_bound)
    right_bound = float(right_bound)
    if left_bound >= right_bound:
        raise ValueError("Peak left bound must be smaller than peak right bound.")

    if left_bound < np.min(E) or right_bound > np.max(E):
        raise ValueError("Peak bounds must lie within the Potential range of the dataset.")

    search_start = _find_closest_index(E, left_bound)
    search_end_inclusive = _find_closest_index(E, right_bound)
    if search_start >= search_end_inclusive:
        raise ValueError("Peak bounds must span at least two distinct data points.")

    search_end = search_end_inclusive + 1
    if search_end - search_start < 3:
        raise ValueError("Peak bounds are too narrow. Choose a wider interval around the desired peak.")

    E_region = E[search_start:search_end]
    I_region = Is[search_start:search_end]

    rough = fit_rough_reference_line_for_apex(E_region, I_region, peak_orientation=peak_orientation, n_slope=1201)
    apex_pick = choose_apex_by_orthogonal_distance(E_region, I_region, rough, peak_orientation=peak_orientation)
    apex_local_idx = apex_pick["local_idx"]
    apex_idx = apex_local_idx + search_start

    return {
        "smooth": Is,
        "apex_idx": apex_idx,
        "left_idx": search_start,
        "right_idx": search_end - 1,
        "requested_left_bound": left_bound,
        "requested_right_bound": right_bound,
        "peak_orientation": peak_orientation,
        "search_start": search_start,
        "search_end": search_end,
        "rough_reference_line": rough["baseline"],
        "rough_reference_m": rough["m"],
        "rough_reference_b": rough["b"],
        "apex_distance_curve": apex_pick["distance"],
        "apex_max_distance": apex_pick["max_distance"],
    }


def build_reference_mask_from_peak_bounds(n, left_idx, right_idx):
    ref_mask = np.zeros(n, dtype=bool)
    if left_idx > 0:
        ref_mask[:left_idx] = True
    if right_idx + 1 < n:
        ref_mask[right_idx + 1:] = True
    return ref_mask


def fit_zero_line_from_outer_points(E, I, ref_mask, peak_orientation="downward", poly_degree=2, eps=None):
    _validate_peak_orientation(peak_orientation)
    E = np.asarray(E)
    I = np.asarray(I)

    if not np.any(ref_mask):
        raise ValueError("No valid outer reference points for zero-line fitting.")

    if eps is None:
        resid0 = I - gaussian_smooth(I, sigma_pts=4)
        noise_est = np.median(np.abs(resid0 - np.median(resid0))) * 1.4826
        eps = max(3 * noise_est, 1e-4)

    E_ref = E[ref_mask]
    I_ref = I[ref_mask]

    max_degree = max(1, min(int(poly_degree), len(E_ref) - 1))
    coeffs = np.polyfit(E_ref, I_ref, max_degree)
    fit_curve = np.polyval(coeffs, E)

    if peak_orientation == "downward":
        delta_shift = np.max(I + eps - fit_curve)
        zero_line = fit_curve + delta_shift
        gap_to_raw = zero_line - I
    else:
        delta_shift = np.min(I - eps - fit_curve)
        zero_line = fit_curve + delta_shift
        gap_to_raw = I - zero_line

    return {
        "coeffs": coeffs,
        "poly_degree": max_degree,
        "fit_curve": fit_curve,
        "delta_shift": delta_shift,
        "zero_line": zero_line,
        "eps": eps,
        "min_gap": np.min(gap_to_raw),
        "gap_to_raw": gap_to_raw,
    }


def swv_workflow(E, I, peak_left_bound, peak_right_bound, peak_orientation="downward", eps=None, poly_degree=2):
    _validate_peak_orientation(peak_orientation)
    peak = find_peak_in_bounds(
        E,
        I,
        left_bound=peak_left_bound,
        right_bound=peak_right_bound,
        peak_orientation=peak_orientation,
    )

    ref_mask = build_reference_mask_from_peak_bounds(
        n=len(I),
        left_idx=peak["left_idx"],
        right_idx=peak["right_idx"],
    )

    zero_fit = fit_zero_line_from_outer_points(
        E,
        I,
        ref_mask=ref_mask,
        peak_orientation=peak_orientation,
        poly_degree=poly_degree,
        eps=eps,
    )

    if peak_orientation == "downward":
        corrected = zero_fit["zero_line"] - I
    else:
        corrected = I - zero_fit["zero_line"]

    return {
        "E": E,
        "I_raw": I,
        "I_smooth": peak["smooth"],
        "peak_orientation": peak_orientation,
        "apex_idx": peak["apex_idx"],
        "left_idx": peak["left_idx"],
        "right_idx": peak["right_idx"],
        "requested_left_bound": peak["requested_left_bound"],
        "requested_right_bound": peak["requested_right_bound"],
        "search_start": peak["search_start"],
        "search_end": peak["search_end"],
        "rough_reference_line": peak["rough_reference_line"],
        "rough_reference_m": peak["rough_reference_m"],
        "rough_reference_b": peak["rough_reference_b"],
        "apex_distance_curve": peak["apex_distance_curve"],
        "apex_max_distance": peak["apex_max_distance"],
        "ref_mask": ref_mask,
        "zero_line": zero_fit["zero_line"],
        "zero_line_fit_curve": zero_fit["fit_curve"],
        "zero_line_coeffs": zero_fit["coeffs"],
        "zero_line_poly_degree": zero_fit["poly_degree"],
        "zero_line_delta_shift": zero_fit["delta_shift"],
        "gap_to_raw": zero_fit["gap_to_raw"],
        "eps": zero_fit["eps"],
        "min_gap": zero_fit["min_gap"],
        "corrected": corrected,
        "no_cross": np.all(zero_fit["gap_to_raw"] > -1e-12),
        "no_touch": np.min(zero_fit["gap_to_raw"]) > 0,
    }

def swv_downward_workflow(E, I, peak_left_bound, peak_right_bound, eps=None, poly_degree=2):
    return swv_workflow(
        E,
        I,
        peak_left_bound=peak_left_bound,
        peak_right_bound=peak_right_bound,
        peak_orientation="downward",
        eps=eps,
        poly_degree=poly_degree,
    )


def swv_upward_workflow(E, I, peak_left_bound, peak_right_bound, eps=None, poly_degree=2):
    return swv_workflow(
        E,
        I,
        peak_left_bound=peak_left_bound,
        peak_right_bound=peak_right_bound,
        peak_orientation="upward",
        eps=eps,
        poly_degree=poly_degree,
    )


def make_plot_raw(E, I):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(E, I * 1e6)
    ax.set_xlabel("Potential")
    ax.set_ylabel("Current (uA)")
    ax.set_title("Step 1 - Raw imported SWV")
    fig.tight_layout()
    return fig


def make_plot_step2(E, I, Is, ref_mask, left_idx, apex_idx, right_idx, search_start, search_end, rough_upper_line):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(E, I * 1e6, alpha=0.45, label="Raw")
    ax.plot(E, Is * 1e6, label="Smoothed")
    ax.axvspan(E[search_start], E[search_end - 1], alpha=0.10, label="Apex search region")
    ax.plot(E[search_start:search_end], rough_upper_line * 1e6, linewidth=2, label="Rough upper line")
    if np.any(ref_mask):
        ax.scatter(E[ref_mask], (I * 1e6)[ref_mask], s=14, label="Outer points")
    ax.axvline(E[left_idx], linestyle="--", label="Left boundary")
    ax.axvline(E[apex_idx], linestyle="--", label="Apex")
    ax.axvline(E[right_idx], linestyle="--", label="Right boundary")
    ax.set_xlabel("Potential")
    ax.set_ylabel("Current (uA)")
    ax.set_title("Step 2 - Peak detection")
    ax.legend()
    fig.tight_layout()
    return fig


def make_plot_zero_line(E, I, fit_curve, zero_line, ref_mask, left_idx, apex_idx, right_idx):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(E, I * 1e6, label="Raw SWV")
    ax.plot(E, fit_curve * 1e6, label="Polynomial fit")
    ax.plot(E, zero_line * 1e6, label="Final zero line")
    if np.any(ref_mask):
        ax.scatter(E[ref_mask], (I * 1e6)[ref_mask], s=14, label="Outer points")
    ax.axvline(E[left_idx], linestyle="--", label="Left boundary")
    ax.axvline(E[apex_idx], linestyle="--", label="Apex")
    ax.axvline(E[right_idx], linestyle="--", label="Right boundary")
    ax.set_xlabel("Potential")
    ax.set_ylabel("Current (uA)")
    ax.set_title("Step 3 - Zero-line reconstruction")
    ax.legend()
    fig.tight_layout()
    return fig


def make_plot_corrected(E, corrected, left_idx, right_idx):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(E, corrected * 1e6, label="Relative curve")
    ax.axhline(0, linestyle="--", label="Zero line")
    ax.axvline(E[left_idx], linestyle="--")
    ax.axvline(E[right_idx], linestyle="--")
    ax.set_xlabel("Potential")
    ax.set_ylabel("Zero-line-relative current (uA)")
    ax.set_title("Step 4 - Curve relative to zero line")
    ax.legend()
    fig.tight_layout()
    return fig


def build_output_dataframe(result):
    return pd.DataFrame(
        {
            "Potential": result["E"],
            "Current_raw_A": result["I_raw"],
            "Current_raw_uA": result["I_raw"] * 1e6,
            "Current_smoothed_A": result["I_smooth"],
            "Current_smoothed_uA": result["I_smooth"] * 1e6,
            "Zero_line_fit_curve_A": result["zero_line_fit_curve"],
            "Zero_line_fit_curve_uA": result["zero_line_fit_curve"] * 1e6,
            "Zero_line_final_A": result["zero_line"],
            "Zero_line_final_uA": result["zero_line"] * 1e6,
            "Current_relative_to_zero_line_A": result["corrected"],
            "Current_relative_to_zero_line_uA": result["corrected"] * 1e6,
            "Is_reference_point_local_outer": result["ref_mask"],
        }
    )


def figure_to_png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
