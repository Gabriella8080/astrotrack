import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

default_font = "Times New Roman"


def plot_doppler_shifts(
    all_satellite_data,
    f0,
    font_family=default_font,
    figsize=(8, 5),
    marker_color="deeppink",
    time_window=10,
):
    """
    Plot the Doppler shifts for each satellite in separate subplots,
    centered around visibility midpoint.

    Parameters:
    ----------
    all_satellite_data : list[dict]
        Satellite data dictionaries.
    f0 : float
        Transmission frequency [Hz].
    font_family : str
        Font family for plot text.
    figsize : tuple
        Figure size for each subplot.
    marker_color : str
        Color of scatter plot markers.
    time_window : float
        Half-width of x-axis around visibility [min].
    """
    c = 3.0e5  # speed of light in km/s
    for satellite_data in all_satellite_data:
        plt.rcParams["font.family"] = font_family
        plt.figure(figsize=figsize)

        epochs = satellite_data["Epochs"]
        velocity_r = np.array(satellite_data["Velocities"])  # km/s
        elevations = satellite_data["Elevations"]
        tle_line_2 = satellite_data["TLE"][1]
        norad_id = tle_line_2.split()[1]
        theta_max = max(elevations)
        epochs_np = np.array([np.datetime64(e) for e in epochs])
        center_time = epochs_np[len(epochs_np) // 2]
        relative_time = (epochs_np - center_time) / np.timedelta64(1, "m")  # mins
        doppler_shift = (velocity_r / c) * f0

        plt.scatter(
            relative_time,
            doppler_shift / 1e3,
            s=3,
            color=marker_color,
            label=rf"$\theta_{{max}} \approx {theta_max:.0f}^\circ$",
        )

        print_time = (
            center_time.astype("datetime64[s]").tolist().strftime("%Y-%m-%d %H:%M:%S")
        )
        plt.xlabel("Visibility Time (min)")
        plt.ylabel("Doppler Shift (kHz)")
        plt.title(f"NORAD {norad_id} Doppler Shift (Centered at {print_time}):")
        plt.axhline(0, color="black", linestyle="--")
        plt.xlim(-time_window, time_window)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True)
        plt.show()


def check_doppler_resolution(
    all_satellite_data, f0_array, resolution=12_000, experiment="REACH", return_df=False
):
    """
    Check if Doppler shift can be resolved,
    given experiment's frequency resolution.

    Parameters:
    ----------
    all_satellite_data : list[dict]
        Satellite data dictionaries.
    f0_array : array
        Frequencies to test [Hz].
    resolution : float
        Frequency resolution of experiment [Hz].
    experiment_name : str
        Name of experiment.
    """
    c = 3.0e5  # speed of light in km/s
    rows = []

    for satellite_data in all_satellite_data:
        norad_id = satellite_data["TLE"][1].split()[1]
        velocity_r = np.array(satellite_data["Velocities"])
        f_max_needed = (resolution * c) / (np.max(np.abs(velocity_r)) * 1e6)

        row = {"Satellite NORAD ID": norad_id}
        for f0 in f0_array:
            max_doppler_khz = np.max(np.abs((velocity_r / c) * f0)) / 1e3
            row[f"{f0/1e6:.1f} MHz"] = f"""{max_doppler_khz:.2f} kHz | {
                'YES' if max_doppler_khz > resolution/1e3 else 'NO'
                }"""
        row[f"Freq. to Resolve ({experiment})"] = f"{f_max_needed:.0f} MHz"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    if return_df:
        return df
