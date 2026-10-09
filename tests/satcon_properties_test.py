from datetime import datetime

import matplotlib
import pytest

matplotlib.use("Agg")
from astrotrack.satcon_properties import (
    filter_by_norads,
    filter_by_time,
    filter_custom,
    filter_nth,
    get_norads,
    plot_flyover_histogram_by_norad,
    plot_max_elevation_histogram,
    plot_satellite_metric,
)


@pytest.fixture
def sample_data():
    return [
        {
            "TLE": ["", "1 12345U"],
            "Epochs": ["2025-06-12T21:36:22", "2025-06-12T21:40:22"],
            "Elevations": [45, 90],
            "Distances": [1000, 800],
        },
        {
            "TLE": ["", "1 23456U"],
            "Epochs": ["2025-06-12T22:00:00"],
            "Elevations": [5],
            "Distances": [1500],
        },
        {
            "TLE": ["", "1 34567U"],
            "Epochs": [],
            "Elevations": [],
        },
    ]


def test_get_norads(sample_data):
    result = get_norads(sample_data)
    assert result == ["12345U", "23456U", "34567U"]


@pytest.mark.parametrize(
    "min_id,max_id,exact_id,expected",
    [
        (None, None, None, 3),
        (20000, None, None, 2),
        (None, 20000, None, 1),
        (None, None, 23456, 1),
    ],
)
def test_filter_by_norads(sample_data, min_id, max_id, exact_id, expected):
    result = filter_by_norads(
        sample_data,
        min_id=min_id,
        max_id=max_id,
        exact_id=exact_id,
    )
    assert len(result) == expected


def test_filter_by_time(sample_data):
    start = datetime(2025, 6, 12, 21, 30)
    end = datetime(2025, 6, 12, 21, 50)

    result = filter_by_time(sample_data, start, end)

    assert len(result) == 1


@pytest.mark.parametrize(
    "step,offset,expected_ids",
    [
        (2, 0, ["12345U", "34567U"]),
        (2, 1, ["23456U"]),
        (1, 0, ["12345U", "23456U", "34567U"]),
    ],
)
def test_filter_nth(sample_data, step, offset, expected_ids):
    result = filter_nth(sample_data, step=step, offset=offset)
    ids = [sat["TLE"][1].split()[1] for sat in result]
    assert ids == expected_ids


def test_filter_custom(sample_data):
    result = filter_custom(
        sample_data, lambda sat: sat.get("Elevations") and max(sat["Elevations"]) > 20
    )
    assert len(result) == 1


def test_plot_satellite_metric_runs(sample_data):
    plot_satellite_metric(sample_data, variable="Elevations")


def test_plot_max_elevation_histogram_runs(sample_data):
    plot_max_elevation_histogram(sample_data)


def test_plot_flyover_histogram_returns_counter(sample_data):
    result = plot_flyover_histogram_by_norad(sample_data)
    assert isinstance(result, dict)
