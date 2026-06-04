"""
Robust DWT T-offset detection for NeuroKit2.

Problem
-------
nk.ecg_delineate(..., method="dwt") locates the T-wave offset by taking the
FIRST negative wavelet modulus maximum after the T peak
(offset_slope_peaks[0] in _dwt_delineate_tp_onsets_offsets). At the scale
NeuroKit auto-selects for many sampling-rate / heart-rate combinations, that
first maximum is a small wiggle right at the T apex, so the detected offset
collapses onto the T peak (typically <30 ms after it).

Fix
---
Select the LARGEST-magnitude negative modulus maximum in the search window
(the real descending-limb inflection of the T wave) instead of the first one.
Everything else -- the wavelet scale, the 0.3 s HR-adjusted search window, and
the offset_weight=0.4 threshold -- is left exactly as NeuroKit computes it, so
this is a minimal, targeted change to the offset anchor only.

Tested on neurokit2 0.2.13.
"""

import numpy as np
import scipy.signal
from neurokit2.ecg.ecg_delineate import (
    _dwt_compute_multiscales,
    _dwt_resample_points,
    _dwt_delineate_tp_peaks,
    _dwt_adjust_parameters,
)
from neurokit2.signal import signal_resample

_ANALYSIS_FS = 2000  # NeuroKit's internal analysis rate for DWT delineation


def robust_dwt_t_offsets(
    ecg_clean,
    rpeaks,
    fs,
    offset_weight: float = 0.4,
    duration_offset: float = 0.3,
):
    """Recompute T-wave offsets from the DWT, anchored to the largest negative
    modulus maximum rather than the first.

    Parameters
    ----------
    ecg_clean : np.ndarray
        Cleaned ECG (e.g. from nk.ecg_clean).
    rpeaks : np.ndarray
        R-peak sample indices at `fs`.
    fs : float
        Sampling rate of `ecg_clean` / `rpeaks`.

    Returns
    -------
    t_offsets_fs : np.ndarray (float)
        T-offset sample indices at the ORIGINAL `fs` (NaN where undetected),
        aligned 1:1 with `rpeaks`.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)

    ecg2k = signal_resample(ecg_clean, sampling_rate=fs, desired_sampling_rate=_ANALYSIS_FS)
    dwtmatr = _dwt_compute_multiscales(ecg2k, 9)

    rpk2k = _dwt_resample_points(rpeaks, fs, _ANALYSIS_FS)
    tpeaks, _ = _dwt_delineate_tp_peaks(ecg2k, rpk2k, dwtmatr, sampling_rate=_ANALYSIS_FS)

    degree = _dwt_adjust_parameters(rpk2k, _ANALYSIS_FS, target="degree")
    dur = _dwt_adjust_parameters(rpk2k, _ANALYSIS_FS, duration=duration_offset, target="duration")
    scale = 2 + degree  # degree_offset (=2) + HR/fs-adjusted degree, same as NeuroKit
    win = int(dur * _ANALYSIS_FS)

    offsets = []
    for tp in tpeaks:
        if not np.isfinite(tp):
            offsets.append(np.nan)
            continue
        s, e = int(tp), int(tp) + win
        loc = dwtmatr[scale, s:e]
        slope_peaks, _ = scipy.signal.find_peaks(-loc)
        if len(slope_peaks) == 0:
            offsets.append(np.nan)
            continue

        # --- the one change vs NeuroKit: largest negative MM, not the first ---
        pk = slope_peaks[np.argmax(-loc[slope_peaks])]
        # ----------------------------------------------------------------------

        eps = -offset_weight * loc[pk]
        cand = np.where(-loc[pk:] < eps)[0] + pk
        if len(cand) == 0:
            offsets.append(np.nan)
            continue
        offsets.append(cand[0] + s)

    offsets = np.asarray(offsets, dtype=float)
    return np.asarray(
        _dwt_resample_points(offsets, _ANALYSIS_FS, desired_sampling_rate=fs),
        dtype=float,
    )


# ---------------------------------------------------------------------------
# How to splice this into run_qt_analysis_from_df:
#
#   _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=fs, method="dwt")
#
#   # replace NeuroKit's collapsed T-offsets with the robust ones
#   t_offsets = robust_dwt_t_offsets(ecg_clean, rpeaks, fs)
#
#   # use QRS onset (already correct in your repo) as the QT start landmark
#   q_onsets = np.asarray(waves["ECG_R_Onsets"], dtype=float)
#
# Everything downstream (qt_s, qt_ms, QTc) stays the same.
# ---------------------------------------------------------------------------
