import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool as Pool

from tqdm import tqdm

import logging_utils
from data_creation import get_urls
from database_functions import *
from database_utils import *

LOGGER = logging_utils.setup_custom_logger("database_data_addition")
URL_FILE = "added_data_log/urls.parquet"


def add_instruments_from_paths_to_database(dict_paths):
    """
    Add instruments from paths to database. If the instrument is already in the database, it will not be added.
    Parameters
    ----------
    dict_paths : dict
        Dictionary of instrument paths.
    """
    # Add the instruments to the database
    for instrument in dict_paths.keys():
        if instrument not in get_table_names_sql():
            for path in dict_paths[instrument]:
                # Try to add the instrument to the database
                # Sometimes it fails because the file is corrupted. In that case, try the next file and break if it works
                try:
                    add_instrument_from_path_to_database(path)
                except Exception as e:
                    LOGGER.error(f"Error adding instrument {instrument}: {e}")
                else:
                    break


def add_specs_from_paths_to_database(urls, chunk_size, cpu_count, replace=False):
    partial_f = partial(add_spec_from_path_to_database, replace=replace)
    with mp.Pool(cpu_count) as pool:
        pool.map_async(
            partial_f,
            tqdm(urls, total=len(urls)),
            chunksize=chunk_size,
        )

        # Wait for all tasks to complete
        pool.close()
        pool.join()


def check_difference_between_two_reports(current_status, previous_status):
    """
    Check the difference between two reports and return the difference.
    Parameters
    ----------
    current_status : pd.DataFrame
        pd.DataFrame containing the current status of the database.
    previous_status : pd.DataFrame
        pd.DataFrame containing the previous status of the database.

    Returns
    -------
    pd.DataFrame
        pd.DataFrame containing the changed files
    """

    # Get the difference between the two reports
    diff = current_status.merge(
        previous_status, how="outer", indicator=True, on=["url", "date"]
    )
    diff = diff[diff["_merge"] != "both"]
    diff = diff.drop(columns=["_merge"])
    return diff


def add_and_check_data_to_database(
    instrument_substring, chunk_size, cpu_count, days_to_observe
):
    # Check data for today and yesterday to create the database.
    LOGGER.info("Checking data for today to create the database.")
    today = datetime.today().date()
    current_status = get_urls(
        today - timedelta(days=days_to_observe), today, instrument_substring
    )
    current_status = pd.DataFrame(current_status)
    # Create the data_today folder if it does not exist
    if not os.path.exists(URL_FILE.split("/")[0]):
        os.makedirs(URL_FILE.split("/")[0])
    # Check if the file exists
    if os.path.exists(URL_FILE):
        previous_status = pd.read_parquet(URL_FILE)
        previous_status = previous_status[
            previous_status["date"].dt.date >= today - timedelta(days=days_to_observe)
        ]
        # Get diff between the new urls and the old ones
    else:
        previous_status = pd.DataFrame(columns=current_status.columns)
    # Get the urls that are not in the already_added_urls
    to_add = check_difference_between_two_reports(current_status, previous_status)

    if len(to_add) == 0:
        LOGGER.info(f"No new data to add between today and {days_to_observe} days ago.")
        return

    dict_paths = create_dict_of_instrument_paths(to_add["url"].to_list())
    LOGGER.info(f"Found {len(dict_paths)} to add in the last {days_to_observe} days.")
    # Add the instruments to the database
    add_instruments_from_paths_to_database(dict_paths)

    # Add the data to the database
    add_specs_from_paths_to_database(
        to_add["url"].to_list(), chunk_size, cpu_count, replace=True
    )

    # Save all the added urls
    df = pd.concat([current_status, to_add])
    df.to_parquet(URL_FILE, index=False)


def main(
    start_date: datetime.date,
    instrument_substring: str,
    chunk_size: int,
    cpu_count: int,
    days_to_observe: int,
) -> None:
    """
    Add instrument data to a database.

    Parameters
    ----------
    start_date : datetime.date
        The starting date for adding instrument data to the database.
    instrument_substring : str
        A substring to match instrument names with.
    chunk_size : int
        The number of instrument data files to add to the database at once.
    cpu_count : int
        The number of CPU cores to use when adding instrument data to the database.

    Returns
    -------
    None

    Notes
    -----
    This function iteratively adds instrument data to a database. First, it creates a database using the data for
    the current day and the previous day. Then, it adds the data to the database. Next, it iterates over all tables in
    the database, and for each table it adds data for the previous `days_chunk_size` days. The function stops when no
    new data is added to the database.

    Examples
    --------
    To add instrument data to a database for instruments containing the substring 'ALASKA-COHOE' starting from
    January 1st, 2023 with a days chunk size of 30, a chunk size of 100, and 8 CPU cores, you could run:

    >>> start_date = datetime.date(2023, 1, 1)
    >>> instrument_substring = 'ALASKA-COHOE'
    >>> days_chunk_size = 30
    >>> chunk_size = 100
    >>> cpu_count = 8
    >>> days_to_observe = 14
    >>> main(start_date, instrument_substring, days_chunk_size, chunk_size, cpu_count, days_to_observe)
    """
    # Add data for today and yesterday to create the database and add new instruments added today.
    add_and_check_data_to_database(
        instrument_substring, chunk_size, cpu_count, days_to_observe
    )
    # Create a list of dates to add to the database
    dates_to_add = pd.date_range(
        start_date, datetime.today().date(), freq="D", inclusive="left"
    )
    for table in (
        get_table_names_sql() if not instrument_substring else instrument_substring
    ):
        LOGGER.info(f"Checking data for {table}.")
        try:
            # Get distinct dates in the database
            dates_in_db = get_distinct_datetime_from_table(table)

            # Get difference of dates
            dates_to_add = np.setdiff1d(dates_to_add, dates_in_db)
            LOGGER.info(
                f"Found {len(dates_to_add)} new dates to add to {table}. Example: {dates_to_add[0]}"
            )

            # Get name of the fits file
            glob_pattern = [reverse_extract_instrument_name(table).lower()]

            # Add the data to the database
            for date in tqdm(
                dates_to_add,
                total=len(dates_to_add),
                desc=f"Adding data for {table}",
            ):
                # Get the urls
                urls = get_urls(
                    date,
                    date,
                    glob_pattern,
                )
                add_specs_from_paths_to_database(urls, chunk_size, cpu_count)
            # Check if new data is added
            add_and_check_data_to_database(
                instrument_substring, chunk_size, cpu_count, days_to_observe
            )
        except ImportError as e:
            LOGGER.error(f"Error adding data to {table}: {e}")


if __name__ == "__main__":
    ## Example:
    # python continiously_add_data_to_database.py --start_date 2023-03-01
    # --instrument_substring "Australia-ASSA, Arecibo-Observatory, HUMAIN, SWISS-Landschlacht, ALASKA-COHOE"
    # The parameter for the multiprocessing is optimized via cprofiler.
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        default=(datetime.today().date() - timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--instrument_substring",
        type=str,
        default="None",
        help="Instrument glob pattern. Default is 'None', which means all instruments currently in the database and every newly added instrument \
        (added in the last two days). Accepts also a list of instruments, e.g. 'Australia-ASSA, Arecibo-Observatory, HUMAIN, SWISS-Landschlacht, ALASKA-COHOE' \
        If you pass a List, only those instruments are updated and the ones added in the last two days are added.",
    )
    parser.add_argument(
        "--days_to_observe",
        type=int,
        default=14,
        help="Number of days to observe for new or changing data. Default is 14.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Chunk size for multiprocessing. Default is 10.",
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=os.cpu_count(),
        help="Number of CPUs to use. Default is all available CPUs.",
    )
    args = parser.parse_args()
    # Update date to datetime
    args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    # Update instrument glob pattern to all if needed or convert to list
    args.instrument_substring = (
        None if args.instrument_substring == "None" else args.instrument_substring
    )
    if args.instrument_substring is not None and "," in args.instrument_substring:
        args.instrument_substring = args.instrument_substring.split(", ")
    LOGGER.info(f"Adding data from {args.start_date}. Args: {args}")
    try:
        # Main
        main(**vars(args))
    except Exception as e:
        LOGGER.exception(e)
        raise e
    LOGGER.info("Done")
