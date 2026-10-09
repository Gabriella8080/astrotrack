import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from skyfield.api import EarthSatellite, load
from tqdm import tqdm


def lat_lon_to_xyz(re, lat, lon):
    """Convert latitude and longitude (degrees) to XYZ coordinates (km)."""
    x = re * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = re * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = re * np.sin(np.radians(lat))
    return x, y, z


def format_timestamps(ts_array):
    """Convert Skyfield times to UTC string list."""
    return [t.utc_strftime("%Y-%m-%d %H:%M:%S") for t in ts_array]


def update_frame(frame, trajectories, plots, timestamps, timestamp_text):
    """Frame update function for FuncAnimation."""
    for i, scatter in enumerate(plots):
        pos = trajectories[i]
        scatter._offsets3d = ([pos[0, frame]], [pos[1, frame]], [pos[2, frame]])
    timestamp_text.set_text(timestamps[frame])
    return plots


def animate_trajectories(
    all_satellite_data,
    elev: float = 30,
    azim: float = 300,
    ref_points=None,
    duration_hours: float = 4,
    step_seconds: int = 60,
    start_time=None,
    output_dir=".",
    filename_prefix="animated_satellites",
    font_family="Times New Roman",
):
    """
    Animate 3D satellite trajectories around Earth and save as GIF.

    Parameters:
    ----------
    all_satellite_data : list[dict]
        Satellite data dictionaries (from preprocess module).
    elev, azim : float
        3D view angles for the animation.
    ref_points : list[tuple[str, float, float]]
        Optional list of (label, lat, lon) reference locations.
    duration_hours : float
        Duration of animation window.
    step_seconds : int
        Time step between frames.
    start_time : datetime
        Start time of animation.
    output_dir : str
        Directory to save animation file.
    filename_prefix : str
        Prefix for output file name.
    font_family : str
        Font family for plot text.
    """
    plt.rcParams["font.family"] = font_family

    ts = load.timescale()
    re = 6378.0  # Earth radius (km)
    theta = np.linspace(0, 2 * np.pi, 201)
    cth, sth, zth = np.cos(theta), np.sin(theta), np.zeros_like(theta)
    lon_lines, lat_lines = [], []
    lon0 = re * np.vstack((cth, zth, sth))
    for phi in np.radians(np.arange(0, 180, 15)):
        cph, sph = np.cos(phi), np.sin(phi)
        cph, sph = [f(phi) for f in [np.cos, np.sin]]
        lon = np.vstack(
            (lon0[0] * cph - lon0[1] * sph, lon0[1] * cph + lon0[0] * sph, lon0[2])
        )
        lon_lines.append(lon)

    for phi in np.radians(np.arange(-75, 90, 15)):
        cph, sph = [f(phi) for f in [np.cos, np.sin]]
        lat = re * np.vstack((cth * cph, sth * cph, zth + sph))
        lat_lines.append(lat)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for x, y, z in lon_lines + lat_lines:
        ax.plot(x, y, z, "-k", lw=0.5, alpha=0.4)

    ref_handles = []

    if ref_points:
        for label, lat, lon in ref_points:
            x, y, z = lat_lon_to_xyz(re, lat, lon)
            h = ax.scatter(x, y, z, s=80, alpha=0.7, label=label)
            ref_handles.append(h)

    plots, trajectories = [], []
    sat_handles = []
    if start_time is not None:
        if isinstance(start_time, datetime):
            start_dt = start_time
        else:
            raise TypeError("start_time must be a datetime.datetime object.")
    else:
        first_sat = all_satellite_data[0]
        start_dt = datetime.fromisoformat(str(first_sat["Epochs"][0]))

    orbit_duration = np.arange(0, duration_hours * 3600, step_seconds)
    time_of_orbit = ts.utc(
        start_dt.year,
        start_dt.month,
        start_dt.day,
        start_dt.hour,
        start_dt.minute,
        start_dt.second + orbit_duration,
    )

    for sat_data in all_satellite_data:
        L1, L2 = sat_data["TLE"]
        sat = EarthSatellite(L1, L2, ts=ts)
        pos = sat.at(time_of_orbit).position.km
        scatter = ax.scatter(
            [pos[0, 0]], [pos[1, 0]], [pos[2, 0]], label=L2.split()[1], s=15
        )
        plots.append(scatter)
        trajectories.append(pos)
        sat_handles.append(scatter)

    timestamps = format_timestamps(time_of_orbit)
    timestamp_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

    ax.view_init(elev, azim)
    ax.set_title(f"Trajectory Animation:\n(elev={elev}°, azim={azim}°)")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")

    max_items = 10
    legend_handles = []
    legend_labels = []
    for h in ref_handles:
        legend_handles.append(h)
        legend_labels.append(h.get_label())
    for h in sat_handles[:max_items]:
        legend_handles.append(h)
        legend_labels.append(h.get_label())
    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        fontsize="small",
        ncol=2,
    )

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename_prefix}_e{elev}_a{azim}.gif")
    writer = PillowWriter(fps=10)
    with tqdm(total=len(timestamps), desc="Rendering animation", unit="frame") as pbar:
        with writer.saving(fig, filepath, dpi=80):
            for i in range(len(timestamps)):
                update_frame(i, trajectories, plots, timestamps, timestamp_text)
                writer.grab_frame()
                pbar.update(1)
    plt.close()
    print(f"Animation saved: {filepath}")
