import datetime
import os

import numpy as np
import pandas as pd
import pytest

from data_creation import (
    check_difference_between_two_reports,
    download_ecallisto_files,
    extract_date_from_path,
)
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


class TestDataCreation:
    def setup(self):
        self.date = datetime.datetime(2021, 1, 1)

    def test_file_download(self, instruments=["ALASKA-COHOE"], dir="test_data"):
        """Test that the file download works."""
        download_ecallisto_files(
            start_date=self.date,
            end_date=self.date,
            instrument=instruments,
            dir=dir,
        )
        assert len(os.listdir(dir)) > 0
        for file in os.listdir(os.path.join(dir, self.date.strftime("%Y/%m/%d"))):
            assert any([instrument in file for instrument in instruments])

    def test_globbing(self, dir="test_data"):
        file_paths = glob_files(
            dir,
            start_date=self.date,
            end_date=self.date,
        )
        assert len(file_paths) == 48

    def test_extraction_of_instrument_names(self, dir="test_data"):
        file_paths = glob_files(
            dir,
            start_date=self.date,
            end_date=self.date,
        )
        instruments = extract_separate_instruments(file_paths)
        assert len(instruments) == 2
        assert "alaska_cohoe_00" in instruments
        assert "alaska_cohoe_01" in instruments

    def test_dict_of_instruments_paths(self, dir="test_data"):
        file_paths = glob_files(
            dir,
            start_date=self.date,
            end_date=self.date,
        )
        instruments = create_dict_of_instrument_paths(file_paths)
        assert len(instruments) == 2
        assert "alaska_cohoe_00" in instruments
        assert "alaska_cohoe_01" in instruments
        assert len(instruments["alaska_cohoe_00"]) == 24
        assert len(instruments["alaska_cohoe_01"]) == 24

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


def test_check_difference_between_two_reports_same_reports():
    current_status = pd.DataFrame(
        {
            "file_name": ["file1.txt", "file2.txt", "file3.txt"],
            "url": [
                "http://example.com/file1.txt",
                "http://example.com/file2.txt",
                "http://example.com/file3.txt",
            ],
            "date_changed": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "date": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "size": [10, 20, 30],
        }
    )
    previous_status = current_status.copy()

    result = check_difference_between_two_reports(current_status, previous_status)

    assert result.empty


def test_check_difference_between_two_reports_different_reports():
    current_status = pd.DataFrame(
        {
            "file_name": ["file1.txt", "file2.txt", "file3.txt"],
            "url": [
                "http://example.com/file1.txt",
                "http://example.com/file2.txt",
                "http://example.com/file3.txt",
            ],
            "date_changed": ["2022-01-02", "2022-01-01", "2022-01-01"],
            "date": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "size": [12, 20, 30],
        }
    )
    previous_status = pd.DataFrame(
        {
            "file_name": ["file1.txt", "file2.txt", "file3.txt"],
            "url": [
                "http://example.com/file1.txt",
                "http://example.com/file2.txt",
                "http://example.com/file3.txt",
            ],
            "date_changed": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "date": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "size": [10, 20, 30],
        }
    )

    result = check_difference_between_two_reports(current_status, previous_status)

    expected_result = pd.DataFrame(
        {
            "file_name": ["file1.txt"],
            "url": ["http://example.com/file1.txt"],
            "date_changed": ["2022-01-02"],
            "date": ["2022-01-01"],
            "size": [12],
        }
    )

    pd.testing.assert_frame_equal(result, expected_result)


def test_check_difference_between_two_reports_one_report_empty():
    current_status = pd.DataFrame(
        {
            "file_name": ["file1.txt", "file2.txt", "file3.txt"],
            "url": [
                "http://example.com/file1.txt",
                "http://example.com/file2.txt",
                "http://example.com/file3.txt",
            ],
            "date_changed": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "date": ["2022-01-01", "2022-01-01", "2022-01-01"],
            "size": [10, 20, 30],
        }
    )

    previous_status = pd.DataFrame(
        {"file_name": [], "url": [], "date_changed": [], "date": [], "size": []}
    )
    result = check_difference_between_two_reports(current_status, previous_status)

    assert result.equals(current_status)
