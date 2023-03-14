import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from radiospectra.sources import CallistoSpectrogram


def int_time_delta_to_pd_timedelta(int_time_delta):
    return pd.Timedelta(seconds=int_time_delta)


def spec_time_to_pd_datetime(spec):
    start_time = pd.to_datetime(spec.start)
    time_axis = spec.time_axis
    time_frequency = spec.time_axis[1] - spec.time_axis[0]
    datetime_range = pd.date_range(
        start_time,
        periods=len(time_axis),
        freq=int_time_delta_to_pd_timedelta(time_frequency),
    )
    return datetime_range


def spec_to_pd_dataframe(spec, kwargs=None):
    df = pd.DataFrame(
        spec.data.T,
        index=spec_time_to_pd_datetime(spec),
        columns=[float(x) for x in spec.freq_axis],
    )
    # Add all the header information to the dataframe
    for key, value in spec.header.items():
        df.attrs[key] = value
    # Add the sunflare type to the dataframe from kwargs
    if kwargs is not None:
        for key, value in kwargs.items():
            df.attrs[key] = value
    return df


def change_resolution_over_freq(
    spectogram, freq_resolution, interpolation=cv2.INTER_LINEAR
):
    """
    Change the resolution of the spectogram over frequency (y-axis)
    :param spectogram: The spectogram to change the resolution of.
    :param resolution: The new resolution.
    :return: The spectogram with the new resolution.
    """
    index = spectogram.index
    columns = spectogram.columns

    spectogram_resized = cv2.resize(
        spectogram.values,
        (freq_resolution, spectogram.shape[0]),
        interpolation=interpolation,
    )

    freq_range = cv2.resize(
        columns.astype(float).values.reshape(-1, 1),
        (1, freq_resolution),
        interpolation=interpolation,
    ).reshape(-1)

    df = pd.DataFrame(
        spectogram_resized, index=index, columns=freq_range.astype(object)
    )
    for key, value in spectogram.attrs.items():
        df.attrs[key] = value

    return df


def change_resolution_over_time(
    spectogram, time_resolution, interpolation=cv2.INTER_LINEAR
):
    """
    Change the resolution of the spectogram over time (x-axis)
    :param spectogram: The spectogram to change the resolution of.
    :param resolution: The new resolution.
    :return: The spectogram with the new resolution.
    """
    index = spectogram.index
    columns = spectogram.columns

    spectogram_resized = cv2.resize(
        spectogram.values,
        (spectogram.shape[1], time_resolution),
        interpolation=interpolation,
    )

    date_range = pd.date_range(start=index[0], end=index[-1], periods=time_resolution)

    df = pd.DataFrame(spectogram_resized, index=date_range, columns=columns)
    for key, value in spectogram.attrs.items():
        df.attrs[key] = value

    return df


def plot_spectogram(spectogram, sunflare_type=None):
    spectogram = spectogram.copy()
    spectogram.index = spectogram.index.strftime("%Y-%m-%d %H:%M:%S")
    spectogram.columns = [f"{col:.2f}" for col in spectogram.columns]
    ax = sns.heatmap(spectogram.T, cmap="viridis", cbar_kws={"label": "Flux (W/m^2)"})
    title = f'Spectogram {spectogram.attrs["instrument"]}'
    if sunflare_type is not None:
        title += f". Sunflaretype: {sunflare_type}"
    plt.title(title)
    plt.ylabel("Frequency (MHz)")
    plt.xlabel("Time UTC")
    plt.show()


def download_spectogram(df_row, mid_time, duration, subtract_background=True):
    spec = CallistoSpectrogram.from_range(
        df_row["instruments"],
        mid_time - duration / 2,
        mid_time + duration / 2,
        exact=True,
    )
    spec = spec.remove_border()
    spec = spec.elimwrongchannels()
    if subtract_background:
        spec = spec.subtract_bg()
    spec = spec_to_pd_dataframe(spec, df_row)
    spec = spec[mid_time - duration / 2 : mid_time + duration / 2]
    return spec


def download_spectogram_from_df_row(df_row, duration):
    mid_time = (
        df_row["datetime_start"]
        + (df_row["datetime_end"] - df_row["datetime_start"]) / 2
    )
    return download_spectogram(df_row, mid_time, duration)


def masked_spectogram_to_array(spectogram):
    """
    Converts a masked spectogram to an array by removing all masked values.
    """
    # Get row with no masked values
    idxs = np.where(~np.any(np.ma.getmaskarray(spectogram.data), axis=1))[0]
    # Keep only frequencies with no masked values
    spectogram.freq_axis = spectogram.freq_axis[idxs]
    # keep only rows in idxs
    data = np.ma.getdata(spectogram.data)
    spectogram.data = data[idxs, :]

    return spectogram
