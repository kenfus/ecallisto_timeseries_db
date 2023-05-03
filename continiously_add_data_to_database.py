import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool as Pool

from tqdm import tqdm

import logging_utils
from data_creation import check_difference_between_two_reports, get_urls, instrument_name_to_regex_pattern
from database_functions import *
from database_utils import *

from astropy.utils.data import clear_download_cache

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

    # Clear the cache to avoid memory issues
    clear_download_cache()


def add_and_check_data_to_database(
    instrument_name , chunk_size, cpu_count, days_to_observe
):
    # Check data for today and yesterday to create the database.
    LOGGER.info("Checking data for today to create the database.")
    today = datetime.today().date()
    instrument_regexr_pattern = instrument_name_to_regex_pattern(instrument_name)
    current_status = get_urls(
        today - timedelta(days=days_to_observe), today, instrument_regexr_pattern 
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
    instrument_name: str,
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
    instrument_name : str
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
    >>> instrument_name = 'ALASKA-COHOE'
    >>> days_chunk_size = 30
    >>> chunk_size = 100
    >>> cpu_count = 8
    >>> days_to_observe = 14
    >>> main(start_date, instrument_name, days_chunk_size, chunk_size, cpu_count, days_to_observe)
    """
    # Add data for today and yesterday to create the database and add new instruments added today.
    add_and_check_data_to_database(
        instrument_name, chunk_size, cpu_count, days_to_observe
    )
    # Create a list of dates to add to the database
    dates_to_add = pd.date_range(
        start_date, datetime.today().date(), freq="D", inclusive="left"
    )
    iterator_tables = get_table_names_sql() if not instrument_name else instrument_name
    for i, table in enumerate(
        iterator_tables
    ):
        LOGGER.info(f"Checking data for {table}. Table number {i + 1} of {len(iterator_tables)}")
        try:
            # Get distinct dates in the database
            dates_in_db = get_distinct_datetime_from_table(table)

            # Get difference of dates
            dates_to_add = np.setdiff1d(dates_to_add, dates_in_db)
            LOGGER.info(
                f"Found {len(dates_to_add)} new dates to add to {table}. Example: {dates_to_add[0]}"
            )

            # Get name of the fits file
            instrument_name = reverse_extract_instrument_name(table)
            LOGGER.info(f"Found instrument name: {instrument_name}")
            instrument_regexr_pattern = instrument_name_to_regex_pattern(instrument_name)
            LOGGER.info(f"Found instrument regex pattern: {instrument_regexr_pattern}")
            # Add the data to the database
            days_added = 0
            for date in tqdm(
                dates_to_add,
                total=len(dates_to_add),
                desc=f"Adding data for {table}",
            ):
                # Get the urls
                status = get_urls(
                    date,
                    date,
                    instrument_regexr_pattern,
                )
                add_specs_from_paths_to_database(status["url"], chunk_size, cpu_count)
                LOGGER.info(f"Added data for {date} to {table}. {days_added} out of {len(dates_to_add)}.")
            # Check if new data is added
            add_and_check_data_to_database(
                instrument_name, chunk_size, cpu_count, days_to_observe
            )
        except Exception as e:
            LOGGER.error(f"Error adding data to {table}: {e}")


if __name__ == "__main__":
    ## Example:
    # python continiously_add_data_to_database.py --start_date 2021-03-01
    # The parameter for the multiprocessing is optimized via cprofiler.
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        default=(datetime.today().date() - timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--instrument_name",
        type=str,
        default="None",
        help="Instrument name. Default is 'None', which means all instruments currently in the database and every newly added instrument \
        (added in the last two days).",
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
    args.instrument_name = (
        None if args.instrument_name == "None" else args.instrument_name
    )
    LOGGER.info(f"Adding data from {args.start_date}. Args: {args}")
    try:
        # Main
        main(**vars(args))
    except Exception as e:
        LOGGER.exception(e)
        raise e
    LOGGER.info("Done")
