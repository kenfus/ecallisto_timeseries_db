from datetime import timedelta

import plotly.express as px


def plot_spectogram(df, instrument_name, start_datetime, end_datetime, size=18):
    fig = px.imshow(df.T)
    fig.update_layout(
        title=f"Spectogram of {instrument_name} from {start_datetime} to {end_datetime}",
        xaxis_title="Datetime",
        yaxis_title="Frequency",
        font=dict(family="Courier New, monospace", size=size, color="#7f7f7f"),
    )

    return fig


def plot_background_image(df, instrument_name, end_time, length, timebucket, size=18):
    # Fill missing hours with NaN

    fig = px.imshow(df.T, aspect="auto")
    fig.update_layout(
        title=f"Background Image of {instrument_name}. Length: {length}. End time: {end_time}. Time bucket: {timebucket}",
        xaxis_title="Time",
        yaxis_title="Frequency",
        font=dict(family="Courier New, monospace", color="#7f7f7f"),
    )

    return fig


def add_burst_to_spectogram(fig, burst_start, burst_end, type, size=18):
    fig.add_vrect(
        x0=burst_start,
        x1=burst_end + timedelta(minutes=1),
        opacity=1,
        annotation_text=f"Type: {type}",
        annotation_position="top left",
        annotation_font_size=size,
    )

    return fig


def plot_spectogram_with_burst(
    df, instrument_name, start_datetime, end_datetime, burst_start, burst_end, size=18
):
    fig = plot_spectogram(df, instrument_name, start_datetime, end_datetime, size)
    fig = add_burst_to_spectogram(fig, burst_start, burst_end, size)

    return fig


def plot_spectogram_with_bursts(
    df, instrument_name, start_datetime, end_datetime, bursts, burst_types=None, size=18
):
    fig = plot_spectogram(df, instrument_name, start_datetime, end_datetime, size)
    if burst_types is None:
        bursts = bursts[bursts["burst_type"] != 0].copy()
    else:
        bursts = bursts[bursts["burst_type"].isin(burst_types)].copy()
    bursts["burst_id"] = (burst_list["burst_type"].diff() != 0).cumsum()
    bursts.reset_index(inplace=True)
    bursts = bursts.groupby("burst_id").agg(
        {"burst_type": "first", "datetime": ["min", "max"]}
    )
    median_time_delta = np.median(np.diff(df.index))
    for index, row in bursts.iterrows():
        fig = add_burst_to_spectogram(
            fig,
            row["datetime"]["min"],
            row["datetime"]["max"] + median_time_delta,
            type=row["burst_type"]["first"],
            size=size,
        )

    return fig
