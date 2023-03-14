import argparse
from datetime import datetime, timedelta

import logging_utils
from data_creation import LOCAL_DATA_FOLDER, download_ecallisto_files
from database_functions import *
from database_utils import *

LOGGER = logging_utils.setup_custom_logger("data_downloader")


def main(start_date, end_date, instrument_glob_pattern="all", dir=LOCAL_DATA_FOLDER):
    download_ecallisto_files(
        instrument=instrument_glob_pattern,
        start_date=start_date,
        end_date=end_date,
        dir=dir,
    )


if __name__ == "__main__":
    ## Example:
    # python download_ecallisto_files.py --start_date 2022-01-01 --end_date 2023-02-20 --instrument_glob_pattern "all"
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date", type=str, default=(datetime.today().date() - timedelta(days=14))
    )
    parser.add_argument("--end_date", type=str, default=datetime.today().date())
    parser.add_argument("--instrument_glob_pattern", type=str, default="all")
    parser.add_argument("--dir", type=str, default=LOCAL_DATA_FOLDER)
    args = parser.parse_args()
    LOGGER.info(
        f"Downloading ecallisto files from {args.start_date} to {args.end_date}. Args: {args}"
    )
    # Update to correct types
    args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    # Main
    main(**vars(args))
