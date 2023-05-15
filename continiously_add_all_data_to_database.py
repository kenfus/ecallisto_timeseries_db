import argparse
import multiprocessing as mp
import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool as Pool
import traceback
from tqdm import tqdm

import logging_utils
from data_creation import get_urls
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
                    LOGGER.info(f"New instrument! Added {instrument} to database.")
                    break
                except Exception as e:
                    LOGGER.error(f"Error adding {instrument} to database: {e}")
                    LOGGER.info(f"Trying next file for {instrument}.")



def add_specs_from_paths_to_database(urls, chunk_size, cpu_count, replace=False):
    partial_f = partial(add_spec_from_path_to_database, replace=replace)
    with mp.Pool(cpu_count) as pool:
        pool.map_async(
            partial_f,
            urls,
            chunksize=chunk_size,
        )

        # Wait for all tasks to complete
        pool.close()
        pool.join()

    # Clear the cache to avoid memory issues
    clear_download_cache()

def main(
    start_date: datetime.date,
    chunk_size: int,
    cpu_count: int
) -> None:
    """
    Add instrument data to a database.

    Parameters
    ----------
    start_date : datetime.date
        The starting date for adding instrument data to the database.
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
    >>> days_chunk_size = 30
    >>> chunk_size = 100
    >>> cpu_count = 8
    >>> main(start_date, days_chunk_size, chunk_size, cpu_count)
    """
    # Create a list of dates to add to the database
    dates_to_add = pd.date_range(
        start_date, datetime.today().date(), freq="D", inclusive="both"
    )
    # Reverse the list of dates to add to the database
    dates_to_add = dates_to_add[::-1]
    
    LOGGER.info(f"Found {len(dates_to_add)} days to add to the database.")
    # Add the data to the database
    days_added = 0
    for date in tqdm(
        dates_to_add,
        total=len(dates_to_add),
        desc=f"Adding data from {start_date}",
    ):
        # Get the urls
        try:
            status = get_urls(
                date,
                date,
            )
            # Check if there are new instruments
            dict_paths = create_dict_of_instrument_paths(status["url"])
            # Add the instruments to the database
            add_instruments_from_paths_to_database(dict_paths)
            # Add the dat a to the database
            add_specs_from_paths_to_database(status["url"], chunk_size, cpu_count)
            LOGGER.info(f"Added data for {date}. {days_added} out of {len(dates_to_add)} done.")
            days_added += 1
        except Exception as e:
            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            tb_str = "".join(tb_str)  # Convert list of strings into a single string
            LOGGER.error(f"Error adding data for {date}: {e}\nTraceback:\n{tb_str}")


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
    LOGGER.info(f"Adding data from {args.start_date}. Args: {args}")
    try:
        # Main
        main(**vars(args))
    except Exception as e:
        LOGGER.exception(e)
        raise e
    LOGGER.info("Done")
