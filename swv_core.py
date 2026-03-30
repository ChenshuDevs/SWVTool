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


def fit_upper_envelope_line(E, I, ref_mask=None, eps=None, n_slope=3501):
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
        b = np.max(I + eps - m * E)
        score = np.mean(m * E_ref + b)
        if best is None or score < best[0]:
            best = (score, m, b)

    _, m_best, b_best = best
    baseline = m_best * E + b_best
    return {"m": m_best, "b": b_best, "baseline": baseline}


def fit_rough_upper_line_for_apex(E_region, I_region, n_slope=1201):
    return fit_upper_envelope_line(
        E_region,
        I_region,
        ref_mask=np.ones_like(I_region, dtype=bool),
        eps=0.0,
        n_slope=n_slope,
    )


def choose_apex_by_orthogonal_distance(E_region, I_region, rough_line):
    m = rough_line["m"]
    b = rough_line["b"]
    denom = np.sqrt(m * m + 1.0)
    distance = (m * E_region - I_region + b) / denom
    local_idx = int(np.argmax(distance))
    return {
        "local_idx": local_idx,
        "distance": distance,
        "max_distance": float(distance[local_idx]),
    }


def find_boundaries_from_rough_line(E_region, Is_region, rough_line, apex_local_idx, amp_frac=0.10):
    R_raw = rough_line - Is_region
    R = moving_average(R_raw, win=FIXED_DERIV_SMOOTH_WIN)
    d2 = second_derivative_nonuniform(E_region, R)

    peak_amp = R[apex_local_idx]
    amp_threshold = amp_frac * peak_amp

    left_idx = 0
    found_left = False
    for i in range(apex_local_idx - 1, 1, -1):
        cond_amp = R[i] <= amp_threshold
        cond_curv = ((d2[i] <= 0 and d2[i - 1] > 0) or abs(d2[i]) < 1e-12)
        if cond_amp and cond_curv:
            left_idx = i
            found_left = True
            break
    if not found_left:
        left_candidates = np.where(R[:apex_local_idx] <= amp_threshold)[0]
        left_idx = int(left_candidates[-1]) if len(left_candidates) > 0 else 0

    right_idx = len(R) - 1
    found_right = False
    for i in range(apex_local_idx + 1, len(R) - 2):
        cond_amp = R[i] <= amp_threshold
        cond_curv = ((d2[i] <= 0 and d2[i + 1] > 0) or abs(d2[i]) < 1e-12)
        if cond_amp and cond_curv:
            right_idx = i
            found_right = True
            break
    if not found_right:
        right_candidates = np.where(R[apex_local_idx:] <= amp_threshold)[0]
        right_idx = int(apex_local_idx + right_candidates[0]) if len(right_candidates) > 0 else len(R) - 1

    return {
        "R_raw": R_raw,
        "R": R,
        "d2": d2,
        "left_local_idx": left_idx,
        "right_local_idx": right_idx,
    }


def find_main_downward_peak_window(E, I, fallback_frac=0.08, edge_exclusion_frac=0.08):
    Is = gaussian_smooth(I, sigma_pts=FIXED_SMOOTH_SIGMA)
    n = len(Is)

    edge_n = int(np.floor(edge_exclusion_frac * n))
    edge_n = max(0, min(edge_n, max(0, n // 2 - 2)))

    search_start = edge_n
    search_end = n - edge_n
    if search_end - search_start < 5:
        search_start = 0
        search_end = n

    E_region = E[search_start:search_end]
    I_region = Is[search_start:search_end]

    rough = fit_rough_upper_line_for_apex(E_region, I_region, n_slope=1201)
    apex_pick = choose_apex_by_orthogonal_distance(E_region, I_region, rough)
    apex_local_idx = apex_pick["local_idx"]
    apex_idx = apex_local_idx + search_start

    boundary = find_boundaries_from_rough_line(
        E_region=E_region,
        Is_region=I_region,
        rough_line=rough["baseline"],
        apex_local_idx=apex_local_idx,
        amp_frac=fallback_frac,
    )

    left_idx = search_start + boundary["left_local_idx"]
    right_idx = search_start + boundary["right_local_idx"]
    left_idx = max(search_start, min(left_idx, apex_idx))
    right_idx = min(search_end - 1, max(right_idx, apex_idx))

    return {
        "smooth": Is,
        "apex_idx": apex_idx,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "search_start": search_start,
        "search_end": search_end,
        "rough_upper_line": rough["baseline"],
        "rough_upper_m": rough["m"],
        "rough_upper_b": rough["b"],
        "apex_distance_curve": apex_pick["distance"],
        "apex_max_distance": apex_pick["max_distance"],
        "rough_detrended_signal_raw": boundary["R_raw"],
        "rough_detrended_signal": boundary["R"],
        "rough_detrended_d2": boundary["d2"],
    }


def build_local_reference_mask(n, search_start, search_end, left_idx, right_idx):
    ref_mask = np.zeros(n, dtype=bool)
    if left_idx > search_start:
        ref_mask[search_start:left_idx] = True
    if right_idx + 1 < search_end:
        ref_mask[right_idx + 1:search_end] = True
    return ref_mask


def fit_zero_line_from_outer_points(E, I, ref_mask, poly_degree=2, eps=None):
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
    delta_shift = np.max(I + eps - fit_curve)
    zero_line = fit_curve + delta_shift
    min_gap = np.min(zero_line - I)

    return {
        "coeffs": coeffs,
        "poly_degree": max_degree,
        "fit_curve": fit_curve,
        "delta_shift": delta_shift,
        "zero_line": zero_line,
        "eps": eps,
        "min_gap": min_gap,
    }


def swv_downward_workflow(E, I, fallback_frac=0.08, edge_exclusion_frac=0.08, eps=None, poly_degree=2):
    peak = find_main_downward_peak_window(
        E,
        I,
        fallback_frac=fallback_frac,
        edge_exclusion_frac=edge_exclusion_frac,
    )

    ref_mask = build_local_reference_mask(
        n=len(I),
        search_start=peak["search_start"],
        search_end=peak["search_end"],
        left_idx=peak["left_idx"],
        right_idx=peak["right_idx"],
    )

    zero_fit = fit_zero_line_from_outer_points(
        E,
        I,
        ref_mask=ref_mask,
        poly_degree=poly_degree,
        eps=eps,
    )

    corrected = zero_fit["zero_line"] - I

    return {
        "E": E,
        "I_raw": I,
        "I_smooth": peak["smooth"],
        "apex_idx": peak["apex_idx"],
        "left_idx": peak["left_idx"],
        "right_idx": peak["right_idx"],
        "search_start": peak["search_start"],
        "search_end": peak["search_end"],
        "rough_upper_line": peak["rough_upper_line"],
        "rough_upper_m": peak["rough_upper_m"],
        "rough_upper_b": peak["rough_upper_b"],
        "apex_distance_curve": peak["apex_distance_curve"],
        "apex_max_distance": peak["apex_max_distance"],
        "rough_detrended_signal_raw": peak["rough_detrended_signal_raw"],
        "rough_detrended_signal": peak["rough_detrended_signal"],
        "rough_detrended_d2": peak["rough_detrended_d2"],
        "ref_mask": ref_mask,
        "zero_line": zero_fit["zero_line"],
        "zero_line_fit_curve": zero_fit["fit_curve"],
        "zero_line_coeffs": zero_fit["coeffs"],
        "zero_line_poly_degree": zero_fit["poly_degree"],
        "zero_line_delta_shift": zero_fit["delta_shift"],
        "eps": zero_fit["eps"],
        "min_gap": zero_fit["min_gap"],
        "corrected": corrected,
        "no_cross": np.all(zero_fit["zero_line"] > I - 1e-12),
        "no_touch": np.min(zero_fit["zero_line"] - I) > 0,
    }


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
