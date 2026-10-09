import matplotlib
import pytest

matplotlib.use("Agg")
from astrotrack.doppler_analysis import check_doppler_resolution


@pytest.fixture
def sample_satellite_data():
    return [
        {
            "TLE": ["", "1 12345U"],
            "Epochs": [
                "2025-06-12T21:36:22",
                "2025-06-12T21:37:22",
                "2025-06-12T21:38:22",
            ],
            "Velocities": [1.0, -2.0, 1.5],  # km/s
            "Elevations": [10, 50, 20],
        },
        {
            "TLE": ["", "1 23456U"],
            "Epochs": [
                "2025-06-12T22:00:00",
                "2025-06-12T22:01:00",
            ],
            "Velocities": [0.5, -0.5],
            "Elevations": [5, 15],
        },
    ]


def test_check_doppler_resolution_returns_df(sample_satellite_data):
    f0_array = [100e6, 150e6]

    df = check_doppler_resolution(
        sample_satellite_data,
        f0_array,
        return_df=True,
    )

    assert df is not None
    assert len(df) == 2


@pytest.mark.parametrize(
    "resolution",
    [1_000, 12_000, 50_000],
)
def test_doppler_resolution_columns(sample_satellite_data, resolution):
    f0_array = [100e6]

    df = check_doppler_resolution(
        sample_satellite_data,
        f0_array,
        resolution=resolution,
        return_df=True,
    )

    # checking column exists:
    assert "100.0 MHz" in df.columns

    for val in df["100.0 MHz"]:
        assert "YES" in val or "NO" in val


def test_zero_velocity_case():
    data = [
        {
            "TLE": ["", "1 99999U"],
            "Epochs": ["2025-06-12T21:36:22"],
            "Velocities": [0.0],
            "Elevations": [10],
        }
    ]

    df = check_doppler_resolution(
        data,
        [100e6],
        return_df=True,
    )

    assert df is not None
