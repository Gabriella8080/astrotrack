from collections import Counter
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from skyfield.api import EarthSatellite, load

# Subset selection:


def get_norads(data):
    return [sat["TLE"][1].split()[1] for sat in data]


def filter_by_norads(data, min_id=None, max_id=None, exact_id=None):
    """
    Return subset of satellites filtered by NORAD ID.

    Parameters:
    ----------
    data : list
        List of satellite dictionaries.
    min_id : int
        Minimum NORAD ID (inclusive).
    max_id : int
        Maximum NORAD ID (inclusive).
    exact_id : int
        One specific NORAD ID exactly.
    """
    subset = []

    for sat in data:
        try:
            norad_str = sat["TLE"][1].split()[1]
            norad = int(norad_str.rstrip("U"))
        except Exception:
            continue
        if exact_id is not None:
            if norad == exact_id:
                subset.append(sat)
        elif (min_id is None or norad >= min_id) and (
            max_id is None or norad <= max_id
        ):
            subset.append(sat)

    return subset


def filter_by_time(data, start_time, end_time):
    """Return subset of satellite flyovers within a specified time window."""
    subset = []
    for sat in data:
        for e in sat.get("Epochs", []):
            try:
                t = datetime.fromisoformat(str(e).replace("Z", ""))
                if start_time <= t <= end_time:
                    subset.append(sat)
                    break
            except Exception:
                continue
    return subset


def filter_nth(data, step=2, offset=0):
    """Return every nth satellite in the dataset."""
    return data[offset::step]


def filter_custom(data, custom):
    """Apply arbitrary filtering function to satellite dataset."""
    return [sat for sat in data if custom(sat)]


# Obtaining satellite set/subset properties:


def plot_satellite_trajectory(
    all_satellite_data,
    elev: float = 30,
    azim: float = 45,
    time=None,
    ref_points=None,
    show_legend=True,
    figsize=(8, 8),
    font_family="Times New Roman",
):
    """
    Plot 3-D satellite trajectories projected around Earth model.

    Parameters:
    ----------
    all_satellite_data : list[dict]
        List of satellite data dictionaries (output of preprocess module).
    elev, azim : float
        Elevation and azimuth angles for user 3-D view.
    time : list
        Skyfield epoch times to compute positions for.
    ref_points : list[tuple[str, float, float]]
        None, optional list of (label, lat, lon) reference locations on Earth.
    show_legend : bool
        Showing legend.
    figsize : tuple
        Matplotlib figure size.
    font_family : str
        Font family for plot text.
    """
    with plt.rc_context({"font.family": font_family}):
        re = 6378.0
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        theta = np.linspace(0, 2 * np.pi, 201)
        lon_grid = re * np.vstack((np.cos(theta), np.zeros_like(theta), np.sin(theta)))
        for phi in np.radians(np.arange(0, 180, 15)):
            cph, sph = np.cos(phi), np.sin(phi)
            x = lon_grid[0] * cph - lon_grid[1] * sph
            y = lon_grid[1] * cph + lon_grid[0] * sph
            z = lon_grid[2]
            ax.plot(x, y, z, "k-", lw=0.5, alpha=0.4)
        for phi in np.radians(np.arange(-75, 90, 15)):
            cph, sph = np.cos(phi), np.sin(phi)
            x = re * np.cos(theta) * cph
            y = re * np.sin(theta) * cph
            z = re * (np.zeros_like(theta) + sph)
            ax.plot(x, y, z, "k-", lw=0.5, alpha=0.4)

        ts = load.timescale()
        for i, sat_data in enumerate(all_satellite_data):
            L1, L2 = sat_data["TLE"]
            sat = EarthSatellite(L1, L2, ts=ts)
            x, y, z = [], [], []
            for t in time:
                pos = sat.at(t).position.km
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])

            norad = L2.split()[1]
            ax.plot(x, y, z, lw=1, alpha=0.8, label=f"{norad}")

        if ref_points:
            for label, lat, lon in ref_points:
                loc = EarthLocation.from_geodetic(lon, lat, height=0 * u.m).to("km")
                ax.scatter(
                    loc.x.value, loc.y.value, loc.z.value, s=100, alpha=0.6, label=label
                )

        if show_legend:
            max_items = 10
            handles, labels = ax.get_legend_handles_labels()
            ref_handles = []
            ref_labels = []
            sat_handles = []
            sat_labels = []
            for handle, label in zip(handles, labels):
                if "sat" in label.lower() or label.isdigit():
                    sat_handles.append(handle)
                    sat_labels.append(label)
                else:
                    ref_handles.append(handle)
                    ref_labels.append(label)
            remaining = max_items - len(ref_handles)
            final_handles = ref_handles + sat_handles[:remaining]
            final_labels = ref_labels + sat_labels[:remaining]
            if final_handles:
                ax.legend(final_handles, final_labels, loc="best", fontsize="small")

        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")
        ax.view_init(elev, azim)
        ax.set_title(
            rf"Satellite Trajectories: (elev={elev}$\degree$, azim={azim}$\degree$)"
        )

        plt.show(block=True)


def plot_flyover_histogram_by_norad(
    all_satellite_data,
    cutoff_norad=None,
    gap_hours: float = 1.0,
    bin_width: int = 500,
    color="lightpink",
    font_family="Times New Roman",
    figsize=(10, 5),
):
    """Plot total satellite passes by NORAD ID (based on consecutive epochs)."""
    with plt.rc_context({"font.family": font_family}):

        def _parse_epoch(e):
            if isinstance(e, datetime):
                return e
            s = str(e).strip()
            if s.endswith("Z"):
                return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
            return datetime.fromisoformat(s)

        passes_per_norad = Counter()

        for sat in all_satellite_data:
            try:
                L2 = sat["TLE"][1]
                norad_id = int(L2.split()[1])
                epochs = sat.get("Epochs", [])
                if not epochs:
                    continue
                times = [_parse_epoch(e) for e in epochs]
            except Exception:
                continue

            pass_count = 1
            for i in range(len(times) - 1):
                dt = (times[i + 1] - times[i]).total_seconds() / 3600
                if dt > gap_hours:
                    pass_count += 1
            passes_per_norad[norad_id] += pass_count

        if len(passes_per_norad) == 0:
            print("No valid satellite passes found.")
            return passes_per_norad

        ids = np.array(list(passes_per_norad.keys()))
        vals = np.array(list(passes_per_norad.values()))
        bins = np.arange(ids.min(), ids.max() + bin_width, bin_width)

        plt.figure(figsize=figsize)
        counts, _, _ = plt.hist(
            ids, bins=bins, weights=vals, color=color, alpha=0.9, edgecolor="none"
        )

        if cutoff_norad:
            plt.axvline(
                cutoff_norad,
                color="red",
                linestyle="--",
                lw=2,
                label=f"NORAD {cutoff_norad}",
            )
            plt.legend()
        plt.xlabel("NORAD ID")
        plt.ylabel("Total Passes")
        plt.title("Distribution of Satellite Flyovers by NORAD ID")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)


def plot_satellite_metric(
    all_satellite_data,
    variable="Elevations",
    threshold=None,
    invert=False,
    font_family="Times New Roman",
    figsize=(12, 6),
):
    """
    Plot time-varying variable (Elevation, Distance, etc.) for all satellite trajectories.

    Parameters:
    ----------
    all_satellite_data : list[dict]
        List of satellite data dictionaries.
    variable : str
        The dictionary key to plot ("Elevations", "Distances").
    threshold : float or None
        Optional threshold.
    invert : bool
        Invert y-axis (for elevation).
    font_family : str
        Font family for plot text.
    figsize : tuple
        Figure size for matplotlib.
    """
    with plt.rc_context({"font.family": font_family}):
        plt.figure(figsize=figsize)
        all_epochs = []
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        ylabel_map = {
            "Elevations": r"Elevation ($\degree$)",
            "Azimuths": r"Azimuth ($\degree$)",
            "Distances": "Distance from Observer (km)",
            "Velocities": "Line-of-sight Velocity (km/s)",
        }

        for i, sat in enumerate(all_satellite_data):
            vals = sat.get(variable, [])
            epochs = sat.get("Epochs", [])
            if not vals or not epochs:
                continue
            arr = np.array(vals, dtype=float)
            if threshold is not None:
                if variable == "Distances" and not np.any(arr < threshold):
                    continue
                if variable == "Elevations" and not np.any(arr > threshold):
                    continue

            try:
                epochs_np = np.array([np.datetime64(e) for e in epochs])
            except Exception:
                continue

            all_epochs.extend(epochs_np)

            norad = (
                sat.get("TLE", ["", ""])[1].split()[1] if sat.get("TLE") else f"sat_{i}"
            )

            color = color_cycle[i % len(color_cycle)]
            # break into segments if large time gaps:
            times_int = epochs_np.astype("datetime64[s]").astype("int")
            gap_threshold = 2700  # 45 mins
            gaps = np.where(np.diff(times_int) > gap_threshold)[0]
            segments = np.split(np.arange(len(arr)), gaps + 1)
            for seg in segments:
                if len(seg) < 2:
                    continue
                plt.plot(
                    epochs_np[seg],
                    arr[seg],
                    color=color,
                    linewidth=1,
                )
            plt.scatter(epochs_np, arr, s=10, color=color, label=f"{norad}")
        handles, labels = plt.gca().get_legend_handles_labels()
        max_items = 10
        if handles:
            plt.legend(
                handles[:max_items],
                labels[:max_items],
                loc="best",
                ncol=2,
                fontsize="small",
                title=f"NORADS ({min(len(labels), max_items)}/{len(labels)})",
            )

        plt.xlabel("Timestamp (UTC)")
        plt.ylabel(ylabel_map.get(variable, variable))
        plt.title(f"{variable} vs Time for Selected Satellites")
        plt.grid(alpha=0.3)

        ax = plt.gca()

        if invert:
            ax.invert_yaxis()

        if all_epochs:
            min_time, max_time = np.min(all_epochs), np.max(all_epochs)
            tick_times = np.linspace(
                min_time.astype("datetime64[s]").astype("int"),
                max_time.astype("datetime64[s]").astype("int"),
                num=15,
            ).astype("datetime64[s]")

            ax.set_xticks(tick_times)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show(block=True)


def plot_max_elevation_histogram(
    all_satellite_data_list,
    font_family="Times New Roman",
    figsize=(10, 5),
    color="lightpink",
    edgecolor="deeppink",
):
    """
    Plot a histogram of maximum elevation angles from all satellites.

    Parameters:
    ----------
    all_satellite_data_list : list
        List of satellite dictionaries, each containing 'Elevations'.
    font_family : str
        Font family for plot text.
    figsize : tuple
        Figure size for matplotlib.
    color : str
        Bar fill color.
    edgecolor : str
        Edge color for histogram bars.
    """
    with plt.rc_context({"font.family": font_family}):
        max_elevations = []
        for sat_data in all_satellite_data_list:
            elevations = sat_data.get("Elevations", [])
            if elevations:
                max_elevations.append(max(elevations))

        bins = np.arange(0, 95, 5)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.figure(figsize=figsize)
        counts, _, _ = plt.hist(
            max_elevations, bins=bins, edgecolor=edgecolor, color=color
        )

        for i, count in enumerate(counts):
            if count > 0:
                plt.text(
                    bin_centers[i], count + 0.5, int(count), ha="center", fontsize=9
                )

        plt.xticks(bins)
        plt.xlim(0, 90)
        plt.xlabel(r"Maximum Elevation Angle ($\degree$)")
        plt.ylabel("Number of Satellites")
        plt.title(
            f"Distribution of Maximum Elevation Angles ({len(all_satellite_data_list)} satellites)"
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show(block=True)
