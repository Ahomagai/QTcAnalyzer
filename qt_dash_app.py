import base64
import io
import math

import neurokit2 as nk
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update


def run_qt_analysis_from_df( df: pd.DataFrame, fs: float = 256.0, rolling_window: int = 5
):

    ecg_raw = df.iloc[:, 0].to_numpy(dtype=float)
    # ECG cleaning 
    ecg_clean = np.asarray(
        nk.ecg_clean(ecg_raw, sampling_rate=fs, method="biosppy"), dtype=float
    )

    # R-peak detection and fix peaks 
    _, rpeaks_info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="neurokit")
    _, rpeaks = nk.signal_fixpeaks(
        rpeaks_info["ECG_R_Peaks"], sampling_rate=fs, method="Kubios"
    )
    rpeaks = np.asarray(rpeaks, dtype=int)

    # Delineate ECG waves (QRS complex and T-wave)
    _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=fs, method="dwt")
    
    # original dwt based t_offset calculation could result in some misidentified (early-labeled) t_offsets, move to peak method
    _, waves2 = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=fs, method = "peak")

    # !----- Important -----!
    # Neurokit2 calculates Q_onset as R_onset, the reasoning being that 
    ## R_Onset == QRS complex onset == Q_onset, change below to reflect ['ECG_R_Onsets'] as q_onset

    q_onsets = np.asarray(waves["ECG_R_Onsets"], dtype=float) 
    
    
    
    t_offsets = np.asarray(waves2["ECG_T_Offsets"], dtype=float)
    
    # TODO: another option is to make a custom version of dwt_t_offest and use that alongside the peak method to run both methods; but this is probably overkill
    # replace NeuroKit's collapsed T-offsets with the robust ones
    # t_offsets = robust_dwt_t_offsets(ecg_clean, rpeaks, fs)
    
    # _, w_peak = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=fs, method="peak")
    # toff_peak = np.asarray(w_peak["ECG_T_Offsets"], dtype=float)
    # toff_dwtfix = robust_dwt_t_offsets(ecg_clean, rpeaks, fs)

    # flag beats where the two offsets disagree by > 40 ms for manual review
    # disagree = np.abs(toff_peak - toff_dwtfix) / fs * 1000 > 40

    # RR intervals (seconds), padded to match beats, as we'll have 1 less RRi than beats
    rr_intervals = np.diff(rpeaks) / fs
    rr_intervals = np.append(rr_intervals, [rr_intervals[-1]])

    # Compute QT interval per beat (seconds and milliseconds)
    qt_s = (t_offsets - q_onsets) / fs
    qt_ms = qt_s * 1000.0

    qt_df = pd.DataFrame(
        {
            "beat": np.arange(len(qt_s)),
            "q_onset_sample": q_onsets,
            "t_offset_sample": t_offsets,
            "q_onset_time_s": q_onsets / fs,
            "t_offset_time_s": t_offsets / fs,
            "qt_s": qt_s,
            "qt_ms": qt_ms,
            "rri": rr_intervals,
        }
    )

    # -------------------------
    # Fig1: Cleaned ECG + Q onset and T Offset markers 

    time = np.arange(len(ecg_clean)) / fs
    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=time,
            y=ecg_clean,
            mode="lines",
            name="ECG (clean)",
            line=dict(color="royalblue", width=1),
        )
    )

    # Q markers
    q_samples_raw = qt_df["q_onset_sample"].to_numpy(dtype=float)
    q_times_raw = qt_df["q_onset_time_s"].to_numpy(dtype=float)
    # deal with NaNs
    valid_q = (
        np.isfinite(q_samples_raw)
        & (q_samples_raw >= 0)
        & (q_samples_raw < len(ecg_clean))
    )
    q_samples = q_samples_raw[valid_q].astype(np.int64)
    q_times = q_times_raw[valid_q]

    fig1.add_trace(
        go.Scatter(
            x=q_times,
            y=ecg_clean[q_samples],
            mode="markers",
            name="Q_onset",
            marker=dict(color="green", size=6, symbol="triangle-up"),
            hovertemplate="Q onset<br>Time: %{x:.3f} s<extra></extra>",
        )
    )

    # T markers, also shows QT interval on hover
    t_samples_raw = qt_df["t_offset_sample"].to_numpy(dtype=float)
    t_times_raw = qt_df["t_offset_time_s"].to_numpy(dtype=float)
    qt_ms_vals = qt_df["qt_ms"].to_numpy(dtype=float)

    valid_t = (
        np.isfinite(t_samples_raw)
        & (t_samples_raw >= 0)
        & (t_samples_raw < len(ecg_clean))
    )
    t_samples = t_samples_raw[valid_t].astype(np.int64)
    t_times = t_times_raw[valid_t]
    qt_ms_for_hover = qt_ms_vals[valid_t]

    fig1.add_trace(
        go.Scatter(
            x=t_times,
            y=ecg_clean[t_samples],
            mode="markers",
            name="T offset",
            marker=dict(color="red", size=6, symbol="triangle-down"),
            customdata=np.round(qt_ms_for_hover, 0),
            hovertemplate=(
                "T offset<br>Time: %{x:.3f} s"
                "<br>QT: %{customdata} ms<extra></extra>"
            ),
        )
    )

    fig1.update_layout(
        title="Beat-to-beat QT intervals (Q onset to T offset)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )

    # Fig2: QTinterval, QTc (Bazett) and rolling means over time 
    beat_time_s = qt_df["t_offset_time_s"].to_numpy(dtype=float)
    qt_ms_vals = qt_df["qt_ms"].to_numpy(dtype=float)
    rri_vals = qt_df["rri"].to_numpy(dtype=float)

    rolling_mean_qtms = (
        pd.Series(qt_ms_vals)
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    QTc_seq = qt_ms_vals / np.sqrt(rri_vals)
    
    meanQTc_seq = np.round(np.nanmean(QTc_seq),3)
    
    QTc_avg_value = np.round(float(np.nanmean(qt_ms_vals) / math.sqrt(np.nanmean(rri_vals))),3) # round values 

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=beat_time_s,
            y=qt_ms_vals,
            mode="markers+lines",
            name="QT (ms)",
            line=dict(color="royalblue", width=1),
            marker=dict(size=5),
            hovertemplate="Time: %{x:.3f} s<br>QT: %{y:.0f} ms<extra></extra>",
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=beat_time_s,
            y=rolling_mean_qtms,
            mode="lines",
            name="Rolling mean QT",
            line=dict(color="red", width=3),
            hovertemplate="Time: %{x:.3f} s<br>Rolling QT: %{y:.0f} ms<extra></extra>",
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=beat_time_s,
            y=QTc_seq,
            mode="lines",
            name="QTc (Bazett, per beat)",
            line=dict(color="green", width=2),
            hovertemplate="Time: %{x:.3f} s<br>QTc: %{y:.0f} ms<extra></extra>",
        )
    )


    fig2.update_layout(
        title=f"Sequential QTc = {meanQTc_seq} | \n  Trial QTc mean = {QTc_avg_value}",
        xaxis_title="Time (s)",
        yaxis_title="QT & QTc Intervals (ms)",
        template="plotly_white",
        legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )

    return qt_df, fig1, fig2


# # Dash app
app = Dash(__name__)
app.title = "QT / QTc ECG Dashboard"
server = app.server  # exposed for gunicorn/WSGI on Render


def parse_contents(contents: str, filename: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    else:
        raise ValueError("Unsupported file type. Please upload a CSV file.")


app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "10px"},
    children=[
        html.H2("QT / QTc ECG Analyzer Dashboard"),
        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                # Left: controls and figures
                html.Div(
                    style={"flex": "3", "display": "flex", "flexDirection": "column"},
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "marginBottom": "10px",
                            },
                            children=[
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Upload ECG File ", html.B("Select ECG CSV")]
                                    ),
                                    style={
                                        "width": "50%",
                                        "height": "40px",
                                        "lineHeight": "40px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "cursor": "pointer",
                                    },
                                    multiple=False,
                                ),
                                html.Div(
                                    children=[
                                        html.Label("Sampling rate fs (Hz): "),
                                        dcc.Input(
                                            id="fs-input",
                                            type="number",
                                            value=256,
                                            debounce=True,
                                            style={"width": "100px"},
                                        ),
                                    ]
                                ),
                                html.Button(
                                    "Run analysis",
                                    id="run-button",
                                    n_clicks=0,
                                    style={"height": "40px"},
                                ),
                                html.Div(
                                    id="status-text",
                                    style={"marginLeft": "10px", "color": "red"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="fig1-graph", style={"flex": "1"}),
                        dcc.Graph(id="fig2-graph", style={"flex": "1"}),
                    ],
                ),
                # Right: table
                html.Div(
                    style={
                        "flex": "2",
                        "display": "flex",
                        "flexDirection": "column",
                        "maxHeight": "850px",
                    },
                    children=[
                        html.H4("QT dataframe (qt_df)"),
                        dash_table.DataTable(
                            id="qt-table",
                            columns=[],
                            data=[],
                            page_size=15,
                            style_table={"height": "100%", "overflowY": "auto"},
                            style_cell={"fontSize": 12, "padding": "4px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("fig1-graph", "figure"),
    Output("fig2-graph", "figure"),
    Output("qt-table", "columns"),
    Output("qt-table", "data"),
    Output("status-text", "children"),
    Input("run-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("fs-input", "value"),
    prevent_initial_call=True,
)
def update_output(n_clicks, contents, filename, fs_value):
    if not contents or not filename:
        return no_update, no_update, no_update, no_update, "Please upload a CSV file."

    if fs_value is None or fs_value <= 0:
        return no_update, no_update, no_update, no_update, "Sampling rate fs must be > 0."

    try:
        df = parse_contents(contents, filename)
        qt_df, fig1, fig2 = run_qt_analysis_from_df(df, fs=float(fs_value))

        columns = [{"name": c, "id": c} for c in qt_df.columns]
        data = qt_df.to_dict("records")
        return fig1, fig2, columns, data, ""

    except Exception as e:
        # Return status message, keep existing figures/table
        return no_update, no_update, no_update, no_update, f"Error: {e}"

# # extra t_wave calculation function
# import numpy as np
# import scipy.signal
# from neurokit2.ecg.ecg_delineate import (
#     _dwt_compute_multiscales,
#     _dwt_resample_points,
#     _dwt_delineate_tp_peaks,
#     _dwt_adjust_parameters,
# )
# from neurokit2.signal import signal_resample

# _ANALYSIS_FS = 2000  # NeuroKit's internal analysis rate for DWT delineation


# def robust_dwt_t_offsets(
#     ecg_clean,
#     rpeaks,
#     fs,
#     offset_weight: float = 0.4,
#     duration_offset: float = 0.3,
# ):
#     """Recompute T-wave offsets from the DWT, anchored to the largest negative
#     modulus maximum rather than the first.

#     Parameters
#     ----------
#     ecg_clean : np.ndarray
#         Cleaned ECG (e.g. from nk.ecg_clean).
#     rpeaks : np.ndarray
#         R-peak sample indices at `fs`.
#     fs : float
#         Sampling rate of `ecg_clean` / `rpeaks`.

#     Returns
#     -------
#     t_offsets_fs : np.ndarray (float)
#         T-offset sample indices at the ORIGINAL `fs` (NaN where undetected),
#         aligned 1:1 with `rpeaks`.
#     """
#     rpeaks = np.asarray(rpeaks, dtype=int)

#     ecg2k = signal_resample(ecg_clean, sampling_rate=fs, desired_sampling_rate=_ANALYSIS_FS)
#     dwtmatr = _dwt_compute_multiscales(ecg2k, 9)

#     rpk2k = _dwt_resample_points(rpeaks, fs, _ANALYSIS_FS)
#     tpeaks, _ = _dwt_delineate_tp_peaks(ecg2k, rpk2k, dwtmatr, sampling_rate=_ANALYSIS_FS)

#     degree = _dwt_adjust_parameters(rpk2k, _ANALYSIS_FS, target="degree")
#     dur = _dwt_adjust_parameters(rpk2k, _ANALYSIS_FS, duration=duration_offset, target="duration")
#     scale = 2 + degree  # degree_offset (=2) + HR/fs-adjusted degree, same as NeuroKit
#     win = int(dur * _ANALYSIS_FS)

#     offsets = []
#     for tp in tpeaks:
#         if not np.isfinite(tp):
#             offsets.append(np.nan)
#             continue
#         s, e = int(tp), int(tp) + win
#         loc = dwtmatr[scale, s:e]
#         slope_peaks, _ = scipy.signal.find_peaks(-loc)
#         if len(slope_peaks) == 0:
#             offsets.append(np.nan)
#             continue

#         # --- the one change vs NeuroKit: largest negative MM, not the first ---
#         pk = slope_peaks[np.argmax(-loc[slope_peaks])]
#         # ----------------------------------------------------------------------

#         eps = -offset_weight * loc[pk]
#         cand = np.where(-loc[pk:] < eps)[0] + pk
#         if len(cand) == 0:
#             offsets.append(np.nan)
#             continue
#         offsets.append(cand[0] + s)

#     offsets = np.asarray(offsets, dtype=float)
#     return np.asarray(
#         _dwt_resample_points(offsets, _ANALYSIS_FS, desired_sampling_rate=fs),
#         dtype=float,
#     )
        

if __name__ == "__main__":
    # Local: python qt_dash_app.py, then open http://127.0.0.1:8050/ in a browser
    # On Render, PORT is injected and gunicorn serves `server` directly (this block doesn't run).
    import os

    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

