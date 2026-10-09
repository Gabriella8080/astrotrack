from datetime import datetime

import numpy as np
import pytest

from astrotrack.psd_analysis import (
    build_time_index_map,
    freq_index,
    get_frequency_bin_range,
    hdf5_index,
    iso_to_hms,
)


def mock_satellite(norad="12345"):
    return {
        "TLE": ["", f"2 {norad}"],
        "Epochs": [
            "2025-06-12T21:36:22",
            "2025-06-12T21:37:22",
            "2025-06-12T21:38:22",
        ],
        "Elevations": [10, 50, 20],
        "Distances": [1500, 800, 1200],
    }


@pytest.mark.parametrize(
    "freq,total_bins,bandwidth,expected",
    [
        (0, 100, 200, 0),
        (100, 100, 200, 50),
        (200, 100, 200, 99),
    ],
)
def test_hdf5_index(freq, total_bins, bandwidth, expected):
    idx = hdf5_index(freq, total_bins, bandwidth)
    assert idx == expected


@pytest.mark.parametrize(
    "bin_idx,total_bins,bandwidth,expected",
    [
        (0, 100, 200, 0.0),
        (50, 100, 200, 100.0),
    ],
)
def test_freq_index(bin_idx, total_bins, bandwidth, expected):
    freq = freq_index(bin_idx, total_bins, bandwidth)
    assert np.isclose(freq, expected)


def test_get_frequency_bin_range():
    start, end = get_frequency_bin_range(
        40, 100, total_bins=100, full_bandwidth_mhz=200
    )
    assert start < end


@pytest.mark.parametrize(
    "input_time,expected",
    [
        ("2025-06-12T21:36:22", "21:36:22"),
        ("2025-06-12T21:36:22Z", "21:36:22"),
        (datetime(2025, 6, 12, 21, 36, 22), "21:36:22"),
    ],
)
def test_iso_to_hms(input_time, expected):
    assert iso_to_hms(input_time) == expected


def test_iso_to_hms_invalid():
    with pytest.raises(TypeError):
        iso_to_hms(12345)


def test_build_time_index_map():
    timestamps = ["21:36:22", "21:37:22", "21:38:22"]
    mapping = build_time_index_map(timestamps)

    assert mapping["21:36:22"] == 0
    assert mapping["21:38:22"] == 2
