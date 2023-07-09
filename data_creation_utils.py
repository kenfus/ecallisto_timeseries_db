import datetime
import glob
import logging
import os
from datetime import datetime
from multiprocessing.pool import Pool as Pool

import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger("database_data_addition")

LOCAL_DATA_FOLDER = os.path.join(os.path.abspath(os.sep), "var", "lib", "ecallisto")
FILES_LOCAL_PATH = "/mnt/nas05/data01/radio/2002-20yy_Callisto/"
MIN_FILE_SIZE = 2000  # Minimum file size in bytes, to redownload empty files

def get_paths(
    start_date, end_date, instrument_regexr_pattern=None, dir=FILES_LOCAL_PATH
):
    """
    Get the local paths of files for a given date range and instrument_regexr_pattern.

    Parameters
    ----------
    start_date : pd.Datetime
        The start date.
    end_date : pd.Datetime
        The end date.
    instrument_regexr_pattern : None or list of str
        The instrument_regexr_pattern name. If None, all instruments are considered.
    dir: str
        The base directory where the data files are stored.

    Returns
    -------
    list of str
        The list of paths of files.
    """
    content = {"file_name": [], "path": [], "date": [], "size": [], "date_changed": []}

    for date in tqdm(
        pd.date_range(start_date, end_date, inclusive="both"), desc="fetching paths"
    ):
        year_path = os.path.join(dir, str(date.year))
        month_path = os.path.join(year_path, str(date.month).zfill(2))
        day_path = os.path.join(month_path, str(date.day).zfill(2))

        # construct file pattern
        if instrument_regexr_pattern:
            file_pattern = os.path.join(day_path, f"*{instrument_regexr_pattern}*")
        else:
            file_pattern = os.path.join(day_path, "*")

        for file_path in glob.glob(file_pattern):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_date_changed = datetime.fromtimestamp(os.path.getmtime(file_path))

            content["file_name"].append(file_name)
            content["path"].append(file_path)
            content["date"].append(date)
            content["size"].append(file_size)
            content["date_changed"].append(file_date_changed)

    return content


def extract_date_from_path(path):
    """Extracts the date from a file path.
    Example: /random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20230127_001500_623.fit.gz -> 20230127_001500
    """
    date = path.split("/")[-1].split(".")[0].split("_")
    if (
        len(date[-1]) < 6 or int(date[-1][:1]) > 24
    ):  # Last element is not a timestamp but an ID
        date.pop()
    date = date[-2:]
    date = datetime.strptime("_".join(date), "%Y%m%d_%H%M%S")
    return date
