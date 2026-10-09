from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from astrotrack.preprocess import (
    filter_tles_by_date,
    load_horizon_profile,
    parse_tle_file,
)

test_tle = [
    "1 45184U 20012G   25064.25002315  .00356607  59443-4  10540-2 0  9990",
    "2 45184  53.0352 192.9089 0003887 301.4292 249.9972 15.92081662279489",
]


@pytest.mark.parametrize(
    "file_contents,satcon,expected_len",
    [
        ("\n".join(test_tle), None, 1),
        ("\n".join(["0 STARLINK-TEST", *test_tle]), "STARLINK", 1),
        ("\n".join(["0 ONEWEB-TEST", *test_tle]), "STARLINK", 0),
    ],
)
def test_parse_tle_file(tmp_path, file_contents, satcon, expected_len):
    file = tmp_path / "tle.txt"
    file.write_text(file_contents)

    result = parse_tle_file(str(file), satcon)

    assert isinstance(result, list)
    assert len(result) == expected_len


@pytest.mark.parametrize(
    "target_date",
    [
        datetime(2024, 4, 10),
        datetime(2023, 1, 1),
    ],
)
def test_filter_tles_by_date_runs(target_date):
    """Test if filtering runs without crashing and returns valid structure."""
    tles = [test_tle]

    result = filter_tles_by_date(tles, target_date)

    assert isinstance(result, list)
    assert all(len(tle) == 2 for tle in result)


# Testing load_horizon_profile:
@pytest.mark.parametrize(
    "azi,elev",
    [  # test horizon profile
        ([0, 90, 180], [0, 5, 0]),
        ([0, 180], [10, 10]),
        ([45, 135, 225], [1, 2, 3]),
    ],
)
def test_load_horizon_from_tuple(azi, elev):
    azi_out, elev_out = load_horizon_profile((azi, elev))

    assert np.array_equal(azi_out, np.array(azi))
    assert np.array_equal(elev_out, np.array(elev))


# Checking CSV input for load_horizon_profile:
@pytest.mark.parametrize(
    "columns",
    [
        ("Azimuth", "Elevation"),
        ("azi", "prof"),
        ("Azimuth", "prof"),
    ],
)
def test_load_horizon_from_csv(tmp_path, columns):
    azi_col, elev_col = columns

    file = tmp_path / "horizon.csv"

    df = pd.DataFrame(
        {
            azi_col: [0, 90, 180],
            elev_col: [0, 5, 0],
        }
    )
    df.to_csv(file, index=False)

    azi, elev = load_horizon_profile(str(file))

    assert len(azi) == 3
    assert len(elev) == 3


def test_load_horizon_missing_columns(tmp_path):
    file = tmp_path / "bad.csv"

    df = pd.DataFrame({"wrong": [1, 2, 3]})
    df.to_csv(file, index=False)

    with pytest.raises(ValueError):
        load_horizon_profile(str(file))
