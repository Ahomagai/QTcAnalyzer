import base64
import io
import math
from pathlib import Path
from typing import Tuple

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
        rpeaks_info["ECG_R_Peaks"], sampling_rate=fs, method="Kubois"
    )
    rpeaks = np.asarray(rpeaks, dtype=int)

    # Delineate ECG waves (QRS complex and T-wave)
    _, waves = nk.ecg_delineate(ecg_clean, rpeaks, sampling_rate=fs, method="dwt")

    # !----- Important -----!
    # Neurokit2 calculates Q_onset as R_onset, the reasoning being that 
    ## R_Onset == QRS complex onset == Q_onset, change below to reflect ['ECG_R_Onsets'] as q_onset

    q_onsets = np.asarray(waves["ECG_R_Onsets"], dtype=float) 
    t_offsets = np.asarray(waves["ECG_T_Offsets"], dtype=float)

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
                                        ["Drag and Drop or ", html.B("Select ECG CSV")]
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


if __name__ == "__main__":
    # Run: python qt_dash_app.py, then open http://127.0.0.1:8050/ in a browser
    app.run(debug=True)

