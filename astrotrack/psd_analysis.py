from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def load_hdf5(
    file_path: str,
    spectra_key: str = "ant_spectra",
    timestamps_key: str = "ant_timestamps",
):
    """
    Load observation HDF5 file to return spectra and UTC timestamps.

    Parameters:
    ----------
    file_path : str
        Path to the user's HDF5 file.
    spectra_key : str
        Dataset name for spectra.
    timestamps_key : str
        Dataset name for timestamps.

    Outputs:
    -------
    np.ndarray
        2D array of PSD measurements (time x frequency bins)
    list[str]
        List of 'HH:MM:SS' formatted UTC timestamps
    """
    try:
        with h5py.File(file_path, "r") as f:
            if "observation_data" not in f:
                raise ValueError("HDF5 file missing 'observation_data' group")

            obs_group = f["observation_data"]
            if spectra_key not in obs_group:
                raise ValueError(f"HDF5 file missing spectra dataset: '{spectra_key}'")
            if timestamps_key not in obs_group:
                raise ValueError(
                    f"HDF5 file missing timestamps dataset: '{timestamps_key}'"
                )

            spectra = obs_group[spectra_key][:]
            timestamps = obs_group[timestamps_key][:]
            utc_timestamps = [
                datetime.fromtimestamp(ts[0]).strftime("%H:%M:%S")
                for ts in timestamps
            ]

    except (OSError, KeyError, ValueError) as e:
        raise RuntimeError(f"Error reading HDF5 file: {e}")

    return spectra, utc_timestamps


def hdf5_index(freq_mhz, total_bins, full_bandwidth_mhz=200):
    """
    Convert frequency (MHz) to bin index for spectral data.

    Parameters:
    ----------
    freq_mhz : float
        Single frequency [MHz].
    total_bins : int
        Total number of bins in spectral dataset.
    full_bandwidth_mhz : float
        Total bandwidth of spectrum [MHz].

    Output:
    ------
    int
        Corresponding bin index.
    """
    bin_idx = int(round(freq_mhz / (full_bandwidth_mhz / total_bins)))
    return max(0, min(bin_idx, total_bins - 1))


def freq_index(bin_idx, total_bins, full_bandwidth_mhz=200):
    """
    Convert bin index to true frequency (MHz) of spectral data.

    Parameters:
    ----------
    bin_idx : int
        Bin index to convert.
    total_bins : int
        Total number of bins in spectral dataset.
    full_bandwidth_mhz : float
        Total bandwidth of spectrum [MHz].

    Output:
    -----
    float
        Frequency corresponding to bin index [MHz].
    """
    return bin_idx * (full_bandwidth_mhz / total_bins)


def get_frequency_bin_range(freq_min, freq_max, total_bins, full_bandwidth_mhz=200):
    """
    Compute start and end bin indices for specified frequency range.

    Parameters:
    ----------
    freq_min : float
        Minimum frequency of range [MHz].
    freq_max : float
        Maximum frequency of range [MHz].
    total_bins : int
        Total number of bins in spectral dataset.
    full_bandwidth_mhz : float
        Total bandwidth of spectrum [MHz].

    Output:
    ------
    tuple[int]
        Start and end bin index corresponding
        to freq_min and freq_max.
    """
    return (
        hdf5_index(freq_min, total_bins, full_bandwidth_mhz),
        hdf5_index(freq_max, total_bins, full_bandwidth_mhz),
    )


def iso_to_hms(t):
    """
    Convert datetime/ISO string/timestamp from HDF5 file
    used by REACH collaboration to 'HH:MM:SS' format.
    """
    if isinstance(t, str):
        t = t.rstrip("Z")
        if "T" in t:
            t = t.split("T")[-1]
        return t[:8]

    if isinstance(t, datetime):
        return t.strftime("%H:%M:%S")

    raise TypeError(f"Unsupported time format: {type(t)}")


def build_time_index_map(utc_timestamps):
    """
    Map UTC timeline to indices using normalised HH:MM:SS keys.
    """
    return {iso_to_hms(t): i for i, t in enumerate(utc_timestamps)}


def plot_psd_with_satellite_metric(
    spectra: np.ndarray,
    utc_timestamps: list[str],
    satellite_data: list[dict],
    variable: str = "Elevations",
    bandwidth: float = 200,
    freq_low_mhz: float = 40,
    freq_high_mhz: float = 170,
    v_min: float = 1e14,
    v_max: float = 8e16,
    show_legend: bool = False,
    threshold: float = None,
    vertical_lines: list[str] = None,
    cmap: str = "magma",
):
    """
    Plot PSD waterfall and chosen satellite variable aligned by time.

    Parameters:
    ----------
    spectra : 2D np.ndarray
        PSD measurements (time x frequency bins).
    utc_timestamps : list[str]
        'HH:MM:SS' formatted timestamps matching spectra.
    satellite_data : list of dict
        Preprocessed satellite data.
    variable : str
        Satellite variable to plot ('Elevations', 'Distances').
    freq_low_mhz, freq_high_mhz : float
        Frequency range to plot [MHz].
    v_min, v_max : float
        Color scale for PSD.
    show_legend : bool
        Display NORAD IDs in plot.
    threshold : float
        Filter satellites by variable threshold.
    vertical_lines : list[str]
        Timestamp lines to mark.
    cmap : str
        Colormap for PSD.
    """
    num_timestamps, total_bins = spectra.shape
    bin_start, bin_end = get_frequency_bin_range(
        freq_low_mhz, freq_high_mhz, total_bins, bandwidth
    )
    bin_start, bin_end = int(bin_start), int(bin_end)
    spectra_subset = spectra[:, bin_start:bin_end]
    utc_to_idx = build_time_index_map(utc_timestamps)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [4, 2]}
    )

    cax = fig.add_axes([0.92, 0.105, 0.02, 0.775])

    im = ax1.imshow(
        spectra_subset.T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=v_min,
        vmax=v_max,
        extent=[0, num_timestamps - 1, freq_low_mhz, freq_high_mhz],
    )

    fig.colorbar(im, cax=cax, label="PSD Intensity")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_title(f"PSD ({freq_low_mhz}-{freq_high_mhz} MHz)")

    norads = []
    for sat_data in satellite_data:
        if "TLE" not in sat_data:
            continue
        norad_id = sat_data["TLE"][1].split()[1]
        values = np.array(sat_data[variable])
        times = [iso_to_hms(ep) for ep in sat_data["Epochs"]]
        if threshold is not None:
            if variable == "Distances" and not np.any(values < threshold):
                continue
            if variable == "Elevations" and not np.any(values > threshold):
                continue

        aligned = np.full(num_timestamps, np.nan)
        matched = 0
        for t, v in zip(times, values):
            idx = utc_to_idx.get(t)
            if idx is not None:
                aligned[idx] = v
                matched += 1
        if matched == 0:
            continue
        ax2.plot(np.arange(num_timestamps), aligned, lw=1, label=norad_id)
        norads.append(norad_id)

    if len(norads) == 0:
        print("No satellites flyovers matched HDF5 Observation UTC timeline.")
        return

    if variable == "Distances":
        ax2.invert_yaxis()
    ax2.set_ylabel(variable)
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.set_title(f"Satellite {variable}")
    x_idx = np.linspace(0, num_timestamps - 1, min(30, num_timestamps), dtype=int)
    x_lbl = [utc_timestamps[i] for i in x_idx]
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_lbl, rotation=45, ha="right")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_lbl, rotation=45, ha="right")

    if vertical_lines:
        for t in vertical_lines:
            idx = utc_to_idx.get(iso_to_hms(t))
            if idx is not None:
                ax1.axvline(idx, color="red", ls="--", lw=1)
                ax2.axvline(idx, color="red", ls="--", lw=1)

    if show_legend:
        ax2.legend()

    fig.suptitle(
        f"Flyovers ({len(norads)} satellites matched)",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def plot_psd_satellite_time_series(
    spectra: np.ndarray,
    utc_timestamps: list[str],
    satellite_data: list[dict],
    norad_list: list[str] = None,
    satellite_variable: str = "Elevations",
    R: float = 2000,
    psd_freq_ranges: list[tuple] = None,
    target_freqs_mhz: list[float] = None,
    bandwidth: int = 200,
    vmin: float = 1e14,
    vmax: float = 8e16,
    cmap: str = "magma",
    line_colors: list[str] = None,
    threshold: float = None,
    vertical_lines: list[str] = None,
):
    """
    Plot PSD panels at single/multiple frequency, with corresponding
    time-series, and chosen satellite variable (Elevations, Distances, etc.) aligned by time.

    Parameters:
    ----------
    spectra : np.ndarray
        2D array (time x frequency bins) of PSD measurements.
    utc_timestamps : list[str]
        Time strings in 'HH:MM:SS' format.
    satellite_data : list[dict]
        Preprocessed satellite data.
    norad_list : list[str]
        List of NORAD IDs for satellite panels, if provided.
    satellite_variable : str
        Satellite variable to plot ('Elevations' or 'Distances').
    R : float
        Radial constraint [km].
    psd_freq_ranges : list of tuples
        [(low1, high1), (low2, high2), ...]; if None, full range used [MHz].
    target_freqs_mhz :list of floats
        Frequencies to plot as narrowband time-series; if None, skip [MHz].
    bandwidth : int
        Frequency bandwidth of spectral dataset [MHz].
    vmin, vmax : float
        Color scale limits for PSD.
    cmap : str
        Colormap for PSD.
    line_colors : list of str
        Colors for narrowband time-series,
    threshold : float
        Optional threshold to hide satellites.
    vertical_lines : list[str]
        UTC times to draw vertical dashed lines.
    """
    num_timestamps, total_bins = spectra.shape
    utc_to_idx = build_time_index_map(utc_timestamps)

    if psd_freq_ranges is None:
        psd_freq_ranges = [(0, bandwidth)]

    if norad_list is None:
        norad_list = [
            sat["TLE"][1].split()[1] for sat in satellite_data if "TLE" in sat
        ]

    if len(norad_list) == 0:
        print("No NORAD list available.")
        return

    x = np.arange(num_timestamps)
    n_psd = len(psd_freq_ranges)
    n_sat = len(norad_list)
    fig = plt.figure(figsize=(14, 2 + 0.9 * n_sat))

    # Top Panel: PSD
    top_panel = fig.add_gridspec(
        1 + n_psd,
        1,
        height_ratios=[0.6] + [1.4] * n_psd,
        hspace=0.05,
        top=0.88,
        bottom=0.5,
    )
    ax_time = fig.add_subplot(top_panel[0])
    ax_psd_list = [
        fig.add_subplot(top_panel[i + 1], sharex=ax_time) for i in range(n_psd)
    ]

    # Bottom Panel(s): Satellite Variable
    bottom_panel = fig.add_gridspec(
        n_sat,
        1,
        hspace=0.25,
        top=0.48,
        bottom=0.08,
    )
    axes_sat = [fig.add_subplot(bottom_panel[i], sharex=ax_time) for i in range(n_sat)]

    # Additional Time Series Panel:
    if target_freqs_mhz is not None:
        bin_groups = {}
        for f in np.atleast_1d(target_freqs_mhz):
            b = int(round(f / bandwidth * total_bins))
            b = max(0, min(b, total_bins - 1))
            if b not in bin_groups:
                bin_groups[b] = [f]
            else:
                bin_groups[b].append(f)

        if line_colors is None:
            line_colors = ["deeppink", "magenta", "purple", "violet", "hotpink"]

        for (b, freqs), color in zip(bin_groups.items(), line_colors):
            ts = spectra[:, b]
            label = ", ".join(f"{f:.1f} MHz" for f in freqs)
            ax_time.plot(x, ts, lw=1.6, color=color, label=label)
        ax_time.legend(fontsize=8)
        ax_time.set_ylabel("PSD")
        ax_time.set_title("Narrowband PSD Time-Series")

    for ax, (low, high) in zip(ax_psd_list, psd_freq_ranges):
        start = int(round(low / bandwidth * total_bins))
        end = int(round(high / bandwidth * total_bins))
        end = min(end, total_bins - 1)
        subset = spectra[:, start:end]
        im = ax.imshow(
            subset.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, num_timestamps - 1, low, high],
        )
        ax.set_ylabel("Frequency (MHz)")

    # Colorbar for PSD:
    cax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    fig.colorbar(im, cax=cax, label="PSD Intensity")

    sat_dict = {sat["TLE"][1].split()[1]: sat for sat in satellite_data if "TLE" in sat}

    plotted = []
    for ax, norad in zip(axes_sat, norad_list):
        if norad not in sat_dict:
            ax.text(0.5, 0.5, f"No data {norad}", ha="center")
            ax.set_yticks([])
            continue
        sat = sat_dict[norad]
        values = np.array(sat[satellite_variable])
        times = [iso_to_hms(ep) for ep in sat["Epochs"]]
        aligned = np.full(num_timestamps, np.nan)
        for t, v in zip(times, values):
            idx = utc_to_idx.get(t)
            if idx is not None:
                aligned[idx] = v

        # threshold formatting:
        if threshold is not None:
            if satellite_variable == "Distances" and not np.any(values < threshold):
                ax.set_visible(False)
                continue
            if satellite_variable == "Elevations" and not np.any(values > threshold):
                ax.set_visible(False)
                continue

        if np.any(~np.isnan(aligned)):
            ax.plot(x, aligned, lw=1, color="deeppink")
            plotted.append(norad)
        else:
            ax.text(0.5, 0.5, f"No overlap {norad}", ha="center")
            ax.set_yticks([])
            continue

        ax.text(1.01, 0.5, f"{norad}", transform=ax.transAxes, va="center")

        if satellite_variable == "Distances":
            ax.set_ylim(0, R)
            ax.invert_yaxis()
        elif satellite_variable == "Elevations":
            ax.set_ylim(0, 90)

    if len(plotted) == 0:
        print("No satellites matched timestamps.")
        return

    x_idx = np.linspace(0, num_timestamps - 1, min(30, num_timestamps), dtype=int)
    x_labels = [utc_timestamps[i] for i in x_idx]

    for ax in [ax_time] + ax_psd_list + axes_sat[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    axes_sat[-1].set_xticks(x_idx)
    axes_sat[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes_sat[-1].set_xlabel("Timestamp (UTC)")

    # vertical_lines formatting:
    if vertical_lines:
        for t in vertical_lines:
            idx = utc_to_idx.get(t)
            if idx is not None:
                for ax in [ax_time] + ax_psd_list + axes_sat:
                    ax.axvline(idx, color="red", linestyle="--", lw=1)

    # y-label formatting:
    if satellite_variable == "Distances":
        fig.text(0.08, 0.3, "Distance (km)", rotation="vertical", va="center")
    else:
        fig.text(0.08, 0.3, r"Elevation ($\degree$)", rotation="vertical", va="center")

    fig.suptitle(f"PSD & Satellite {satellite_variable}", fontsize=12)
    plt.show()
