# Download all spectograms with burst in the corresponding folder
import datetime
import logging
import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool as Pool
from typing import Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("database_data_addition")

LOCAL_DATA_FOLDER = os.path.join(os.path.abspath(os.sep), "var", "lib", "ecallisto")
FILES_BASE_URL = "http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"
MIN_FILE_SIZE = 2000  # Minimum file size in bytes, to redownload empty files

# Requests session
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)


def fetch_content(url):
    reqs = session.get(url)
    soup = BeautifulSoup(reqs.text, "html.parser")
    LOGGER.info(f"Fetching content from {url}")
    LOGGER.info(f"Status code: {reqs.status_code}")
    return soup


def extract_content(
    soup,
    substring_must_match,
    substrings_to_include,
):
    """
    Extracts all the content from the given soup object based on the given parameters
    substring_must_match: If specified, only links with the substring_must_match will be extracted. Capitalization is ignored
    substrings_to_exclude: If specified, only links without the given substrings will be extracted

    Returns a list of all the links
    """
    content = []
    # If the substrings_to_include is not a list, make it a list if it is not None
    if not substrings_to_include is None and not isinstance(
        substrings_to_include, list
    ):
        substrings_to_include = [substrings_to_include]
    for link in soup.find_all("a"):
        if substring_must_match in link.get("href"):
            if substrings_to_include is None or any(
                [
                    substring.lower() in link.get("href").lower()
                    for substring in substrings_to_include
                ]
            ):
                content.append(link.get("href"))
    LOGGER.info(
        f"Extracted {len(content)} files with the following substrings: {substrings_to_include} and the following substring must match: {substring_must_match}"
    )
    if len(content) > 0:
        LOGGER.info(f"Example of extracted files: {np.random.choice(content, 3)}")
    else:
        LOGGER.info(f"No files extracted")
    return content


def extract_fit_gz_files(url, instrument_substring, substring_must_match=None):
    """
    Extracts all the .fit.gz files from the given url
    instrument_substring: If specified, only files with the instrument_substring name will be extracted
    substring_must_match: If specified, only files with the given substrings will be extracted

    Returns a list of all the .fit.gz files
    """
    soup = fetch_content(url)
    if substring_must_match is None:
        substring_must_match = ".fit.gz"

    LOGGER.info(
        f"Extracting files with the following substrings: {substring_must_match}"
    )
    return extract_content(
        soup,
        substring_must_match=substring_must_match,
        substrings_to_include=instrument_substring,
    )


def extract_fiz_gz_files_urls(year, month, day, instrument_substring):
    """
    Extracts all the .fit.gz files from the given year, month and day
    instrument_substring: If specified, only files with the instrument_substring name will be extracted

    Returns a list of all the .fit.gz files
    """
    url = f"{FILES_BASE_URL}{year}/{month}/{day}/"
    file_names = extract_fit_gz_files(url, instrument_substring=instrument_substring)
    urls = [url + file_name for file_name in file_names]
    LOGGER.info(f"Extracted {len(urls)} files")
    return urls


def download_ecallisto_file(URL, return_download_path=False, dir=LOCAL_DATA_FOLDER):
    # Split URL to get the file name and add the directory
    year, month, day, filename = URL.split("/")[-4:]
    directory = os.path.join(dir, year, month, day)
    os.makedirs(directory, exist_ok=True)

    # Check if the file already exists
    file_path = os.path.join(directory, filename)
    if (
        not os.path.exists(file_path) or os.path.getsize(file_path) < MIN_FILE_SIZE
    ):  # Check that it is not an empty file (e.g. 404 error)
        # Downloading the file by sending the request to the URL
        req = session.get(URL)
        with open(file_path, "wb") as output_file:
            output_file.write(req.content)
    LOGGER.debug(f"Downloaded file {file_path}")
    # Return path (e.g. for astropy.io.fits.open)
    if return_download_path:
        return file_path


def get_urls(start_date, end_date, instrument_substring) -> list[str]:
    """
    Get the urls of fiz gz files for a given date range and instrument_substring.

    Parameters
    ----------
    start_date : pd.Datetime
        The start date.
    end_date : pd.Datetime
        The end date.
    instrument_substring : None or list of str
        The instrument_substring name. If None, all instruments are considered. Also, the capitalization is ignored.

    Returns
    -------
    list of str
        The list of urls of fiz gz files.
    """

    urls = []
    for date in tqdm(pd.date_range(start_date, end_date), desc="fetching urls"):
        url = extract_fiz_gz_files_urls(
            date.year,
            str(date.month).zfill(2),
            str(date.day).zfill(2),
            instrument_substring=instrument_substring,
        )
        LOGGER.debug(f"extracted {len(url)} files for {date}")
        urls.extend(url)

    return urls


def download_ecallisto_files(
    dir,
    start_date=datetime.today().date() - timedelta(days=1),
    end_date=datetime.today().date(),
    instrument: Union[list, None] = None,
    return_download_paths=False,
):
    """
    Downloads all the eCallisto files from the given start date to the end date.

    Parameters
    ----------
    dir : str
        Directory where the files will be downloaded.
    start_date : datetime, optional
        Start date of the download. Default is the date of yesterday.
    end_date : datetime, optional
        End date of the download. Default is the date of today.
    instrument : list, optional
        If specified, only files containing any of the given instruments (substring) will be downloaded. Default is None (all instruments).
    return_download_paths : bool, optional
        If True, the paths of the downloaded files will be returned. Default is False.

    Returns
    -------
    None or List of str
        None or a list of paths to the downloaded files, depending on return_download_paths.

    Raises
    ------
    AssertionError
        If start_date is greater than end_date.

    Notes
    -----
    The function uses the `extract_fiz_gz_files_urls` and `download_ecallisto_file` functions to download the files. The logging messages will be written to the LOG_FILE.
    """
    assert (
        start_date <= end_date
    ), "Start date should be less than end date and both should be datetime objects"
    if isinstance(instrument, str) and instrument.lower() in ["*", "all"]:
        instrument = None
    LOGGER.info(
        f"Downloading files from {start_date} to {end_date} (instrument: {instrument if instrument else 'all'})"
    )
    urls = get_urls(start_date, end_date, instrument)
    # Create a partial function to pass the dir argument and return_download_path
    fn = partial(
        download_ecallisto_file, return_download_path=return_download_paths, dir=dir
    )
    # Multiprocessing via tqdm
    with Pool() as p:
        r = list(tqdm(p.imap(fn, urls), total=len(urls), desc="Downloading files"))

    if return_download_paths:
        return r


if __name__ == "__main__":
    print(f"Downloading files to {LOCAL_DATA_FOLDER}")
    download_ecallisto_files(
        dir=LOCAL_DATA_FOLDER,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 2),
        instrument="ALASKA",
    )
