# SkyField
# Numerical
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u

# Astropy
from astropy.coordinates import ITRS, AltAz, CartesianRepresentation, EarthLocation
from astropy.time import Time
from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time
from tqdm import tqdm

ts = load.timescale()


def parse_tle_file(tle_file: str, satcon_name: str = None):
    """
    Parse a TLE file and return a list of (line1, line2) tuples,
    supporting 3LEs, TLEs, mixed formats, or single TLE entries.

    Parameters:
    ----------
    tle_file : str
        Path to TLE text file
    satcon_name : str
        Optional constellation name to filter for

    Output:
    ------
    list[tuple[str, str]]
        Parsed TLE pairs ready for processing
    """
    with open(tle_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    tles = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("0 "):
            if i + 2 < len(lines):
                name = line[2:].strip().upper()
                line1 = lines[i + 1]
                line2 = lines[i + 2]
                if (satcon_name is None) or (satcon_name.upper() in name):
                    tles.append((line1, line2))
                i += 3
            else:
                break
        elif line.startswith("1 ") and i + 1 < len(lines):
            line1 = lines[i]
            line2 = lines[i + 1]
            tles.append((line1, line2))
            i += 2
        else:
            i += 1

    if not tles and len(lines) == 2 and lines[0].startswith("1 "):
        tles.append((lines[0], lines[1]))

    return tles


def filter_tles_by_date(satcon_tles, target_date):
    """
    Filter TLEs within plus/mins 2 weeks of target_date.

    Parameters:
    ----------
    satcon_tles : list[tuple])
        List of (line1, line2) TLE strings
    target_date : datetime)
        Observation date

    Output:
    ------
    list[tuple]
        Filtered TLEs valid within p/m 2 weeks of target_date
    """
    start_date = target_date - timedelta(weeks=2)
    end_date = target_date + timedelta(weeks=2)
    filtered_tles = []
    for tle in satcon_tles:
        sat = EarthSatellite(tle[0], tle[1], ts=ts)
        tle_epoch = sat.epoch.utc_datetime().replace(tzinfo=None)
        if start_date <= tle_epoch <= end_date:
            filtered_tles.append((tle[0], tle[1]))
    return filtered_tles


def compute_sat_properties(sat, location, epoch):
    """
    Calculates properties of satellite trajectory
    relative to observer at epoch.

    Parameters:
    ----------
    sat : EarthSatellite
        Skyfield satellite object
    location : EarthLocation
        Observers ITRS position
    epoch : Time
        Skyfield observation time

    Outputs:
    -------
    tuple
        azimuth [deg], elevation [deg], distance of
        satellite w.r.t observer [km], line-of-sight velocity [km/s]
    """
    sat_at_epoch = sat.at(epoch)
    sat_xyz = sat_at_epoch.position.km
    astropy_time = Time(epoch.utc_iso())

    observer_xyz = np.array([location.x.value, location.y.value, location.z.value])
    displacement = np.array(sat_xyz) - observer_xyz
    distance = np.linalg.norm(displacement)
    unit_displacement = displacement / distance
    relative_velocity = np.array(sat_at_epoch.velocity.km_per_s)
    los_velocity = -np.dot(relative_velocity, unit_displacement)

    altaz_frame = AltAz(obstime=astropy_time, location=location)
    sat_itrs = ITRS(CartesianRepresentation(sat_xyz * u.km), obstime=astropy_time)
    sat_altaz = sat_itrs.transform_to(altaz_frame)
    azimuth = sat_altaz.az.value
    elevation = sat_altaz.alt.value

    return azimuth, elevation, distance, los_velocity


def load_horizon_profile(data):
    """
    Loads horizon profile at observer's location from .csv file or lists.

    Parameters:
    ----------
    data : tuple
        Path to .csv file (azimuth, elevation columns), or
        tuple of two lists (azimuths, elevations)

    Outputs:
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of azimuthal [deg] and corresponding
        elevation [deg] angles
    """
    if isinstance(data, tuple):
        return np.array(data[0]), np.array(data[1])

    csv = pd.read_csv(data)
    azi_label = ["Azi", "Azimuth", "azi", "azimuth"]
    elev_label = ["Prof", "Elevation", "prof", "elevation"]
    azi_col = next((c for c in csv.columns if c in azi_label), None)
    elev_col = next((c for c in csv.columns if c in elev_label), None)

    if azi_col is None or elev_col is None:
        raise ValueError(
            f" Azimuth & Elevation columns in {data} not detected. "
            "Ensure CSV file contains azimuthal and elevation angles [deg]."
        )

    return csv[azi_col].values, csv[elev_col].values


def find_target_sats(
    filtered_tles,
    epochs,
    obs_lat,
    obs_lon,
    azi_list,
    prof_list,
    R,
    obs_height=0,
    start_index=0,
):
    """
    Compute satellites visible above horizon
    within radial constraint at given epochs.

    Parameters:
    ----------
    filtered_tles : list[tuple]
        Filtered TLEs
    epochs : Skyfield
        Array of observation times
    obs_lat : float
        Observer latitude [deg]
    obs_lon : float
        Observer longitude [deg]
    azi_list : list
        Horizon azimuths [deg]
    prof_list : list
        Corresponding Horizon elevations [deg]
    R : float
        Radial constraint [km]
    obs_height : float
        Observer height [km]
    start_index : int
        Starting index of satellite numbering

    Output:
    list [dict]
        Satellite dictionary with TLEs and
        flyover trajectory properties
    """
    sat_data_list = []
    obs_loc = EarthLocation.from_geodetic(obs_lon, obs_lat, height=obs_height * u.m)
    obs_location = obs_loc.to("km")

    for index, tle in enumerate(
        tqdm(filtered_tles, desc="Processing satellites", unit="satellite")
    ):
        sat_trajectory = EarthSatellite(tle[0], tle[1], ts=ts)
        sat_dict = {
            "TLE": (tle[0], tle[1]),
            "Epochs": [],
            "Azimuths": [],
            "Elevations": [],
            "Distances": [],
            "Velocities": [],
        }

        for epoch in epochs:
            satellite_position = sat_trajectory.at(epoch)
            sat_x, sat_y, sat_z = satellite_position.position.km
            distance_squared = (
                (sat_x - obs_location.x.value) ** 2
                + (sat_y - obs_location.y.value) ** 2
                + (sat_z - obs_location.z.value) ** 2
            )
            if distance_squared > R**2:
                continue

            azimuth, elevation, distance, vel = compute_sat_properties(
                sat_trajectory, obs_location, epoch
            )
            azi_index = np.abs(np.array(azi_list) - azimuth).argmin()
            required_prof = prof_list[azi_index]
            if elevation >= required_prof:
                sat_dict["Epochs"].append(epoch.utc_iso())
                sat_dict["Azimuths"].append(azimuth)
                sat_dict["Elevations"].append(elevation)
                sat_dict["Distances"].append(distance)
                sat_dict["Velocities"].append(vel)

        if sat_dict["Epochs"]:
            sat_data_list.append(sat_dict)

    return sat_data_list


def load_satellite_data(
    tle_file,
    target_date,
    obs_len: float,
    traj_res: float,
    obs_lat,
    obs_lon,
    R: float,
    horizon_data,
    satcon: str = None,
    start_index=0,
):
    """
    Main pipeline for processing data about chosen satellite constellation.

    Parameters:
    ----------
    tle_file : str
        Path to TLE plaintext file
    target_date : datetime
        Observation start date
    obs_len : float
        Length of observation [seconds]
    traj_res : float
        Resolution of trajectory samples [seconds]
    obs_lat : float
        Observer latitude [deg]
    obs_lon : float
        Observer longitude [deg]
    R : float
        Radial constraint [km]
    horizon_data : tuple
        Horizon profile of observer
    satcon : str
        Satellite constellation name
    start_index : int
        Starting satellite index, default = 0

    Output:
    ------
    list[dict]
        List of satellite flyover data
    """
    satcon_name = satcon.upper() if satcon else None
    satcon_tles = parse_tle_file(tle_file, satcon_name)

    filtered_tles = filter_tles_by_date(satcon_tles, target_date)
    orbit_duration = np.arange(0, (obs_len + 1), traj_res)
    epochs = ts.utc(
        target_date.year,
        target_date.month,
        target_date.day,
        target_date.hour,
        target_date.minute,
        target_date.second + orbit_duration,
    )

    azi_list, prof_list = load_horizon_profile(horizon_data)
    sat_data = find_target_sats(
        filtered_tles, epochs, obs_lat, obs_lon, azi_list, prof_list, R
    )

    return sat_data
