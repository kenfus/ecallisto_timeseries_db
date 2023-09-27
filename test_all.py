import datetime
import os

import numpy as np
import pandas as pd
import pytest

from data_creation_utils import extract_date_from_path
from database_utils import (
    combine_non_unique_frequency_axis_mean,
    create_dict_of_instrument_paths,
    extract_instrument_name,
    extract_separate_instruments,
    glob_files,
    np_array_to_postgresql_array,
    numbers_list_to_postgresql_columns_meta_data,
    reverse_extract_instrument_name,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("alaska_cohoe_612", "ALASKA-COHOE"),
        ("alaska_cohoe", "ALASKA-COHOE"),
        ("fhn_w_11", "FHN-W"),
        ("station_1234_5678", "STATION-1234"),
    ],
)
def test_reverse_extract_instrument_name(test_input, expected):
    assert reverse_extract_instrument_name(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "/var/lib/ecallisto/2023/01/27/ALASKA-COHOE_20230127_001500_612.fit.gz",
            "alaska_cohoe_612",
        ),
        (
            "/random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20230127_001500_61212.fit.gz",
            "alaska_cohoe_61212",
        ),
        (
            "/random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20230127_001500.fit.gz",
            "alaska_cohoe",
        ),
        (
            "/ran3123öü¨ö23üöeaöd¨üö2¨/ecallisto/2023/01/27/FHN_W_20230127_001500_11.fit.gz",
            "fhn_w_11",
        ),
    ],
)
def test_instrument_name_extraction(test_input, expected):
    assert extract_instrument_name(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            "/var/lib/ecallisto/2023/01/27/ALASKA-COHOE_20230127_001500_612.fit.gz",
            "20230127_001500",
        ),
        (
            "/random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20230127_001500_61212.fit.gz",
            "20230127_001500",
        ),
        (
            "/random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20230127_001500.fit.gz",
            "20230127_001500",
        ),
        (
            "/random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20210127_001113.fit.gz",
            "20210127_001113",
        ),
        (
            "/ran3123öü¨ö23üöeaöd¨üö2¨/ecallisto/2023/01/27/FHN_W_20230127_001500_11.fit.gz",
            "20230127_001500",
        ),
    ],
)
def test_datee_extraction(test_input, expected):
    expected = datetime.datetime.strptime(expected, "%Y%m%d_%H%M%S")
    assert extract_date_from_path(test_input) == expected


@pytest.mark.parametrize(
    "names, types, expected",
    [
        (
            ["test", "test2", "test3"],
            ["int", "float", "varchar"],
            "test int, test2 float, test3 varchar",
        ),
        (
            [3133213, 1, 0.0],
            ["int", "float", "varchar"],
            '"3133213" int, "1" float, "0.0" varchar',
        ),
        (
            [31.3, "testa", 0.013],
            ["int", "varchar", "varchar"],
            '"31.3" int, testa varchar, "0.013" varchar',
        ),
    ],
)
def test_sql_column_creation(names, types, expected):
    assert numbers_list_to_postgresql_columns_meta_data(names, types) == expected

    def test_remove_files(self, dir="test_data"):
        file_paths = glob_files(
            dir,
            start_date=datetime.datetime(2021, 1, 1),
            end_date=datetime.datetime(2021, 1, 1),
        )
        assert len(file_paths) > 0
        for file in file_paths:
            os.remove(file)
        file_paths = glob_files(
            dir,
            start_date=datetime.datetime(2021, 1, 1),
            end_date=datetime.datetime(2021, 1, 1),
        )
        assert len(file_paths) == 0


@pytest.mark.parametrize(
    "input,expected",
    [
        (np.array([[1, 2, 3], [4, 5, 6]]), "(1, 2, 3),(4, 5, 6)"),
        (np.array([[1, 2, 3]]), "(1, 2, 3)"),
    ],
)
def test_np_array_to_postgresql_array(input, expected):
    assert np_array_to_postgresql_array(input) == expected


def test_combine_non_unique_frequency_axis_mean():
    index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5])
    data = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
        ]
    )
    result, unique_idxs = combine_non_unique_frequency_axis_mean(index, data)

    expected_result = np.array(
        [
            [1, 2, 3],
            [(4 + 7) / 2, (5 + 8) / 2, (6 + 9) / 2],
            [(10 + 13 + 16) / 3, (11 + 14 + 17) / 3, (12 + 15 + 18) / 3],
            [19, 20, 21],
            [(22 + 25) / 2, (23 + 26) / 2, (24 + 27) / 2],
        ]
    )
    expected_unique_idxs = np.array([1, 2, 3, 4, 5])

    assert np.array_equal(result, expected_result)
    assert np.array_equal(unique_idxs, expected_unique_idxs)
