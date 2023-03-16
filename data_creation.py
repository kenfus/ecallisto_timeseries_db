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


def extract_date_from_path(path):
    """Extracts the date from a file path.
    Example: /random_2313/ecallisto/2023/01/27/ALASKA_COHOE_20230127_001500_623.fit.gz -> 2023-01-27 00:15:00
    """
    date = path.split("/")[-1].split(".")[0].split("_")
    if (
        len(date[-1]) < 6 or int(date[-1][:1]) > 24
    ):  # Last element is not a timestamp but an ID
        date.pop()
    date = date[-2:]
    date = datetime.strptime("_".join(date), "%Y%m%d_%H%M%S")
    return date


def fetch_content(url):
    reqs = session.get(url)
    soup = BeautifulSoup(reqs.text, "html.parser")
    LOGGER.info(f"Fetching content from {url}")
    LOGGER.info(f"Status code: {reqs.status_code}")
    return soup


def extract_date_size_from_soup(soup, url):
    """
    Extracts the date and size of the file from the soup object
    """
    date = soup.find("a", href=url).parent.find_next_sibling("td").text
    size = (
        soup.find("a", href=url)
        .parent.find_next_sibling("td")
        .find_next_sibling("td")
        .text
    )
    return date, size


def extract_content(
    url,
    instrument_glob_pattern,
    substring_must_match="fit.gz",
    return_date_size=True,
):
    """
    Get the URLs of .fiz.gz files for a given date range and instrument substring.

    Parameters
    ----------
    start_date : pandas.Timestamp
        The start date.
    end_date : pandas.Timestamp
        The end date.
    instrument_substring : {None, list of str}, optional
        The instrument substring name(s). If None, all instruments are considered. Also, the capitalization is ignored.

    Returns
    -------
    list of str
    The list of URLs of .fiz.gz files.

    Example:
    # Extract all files that include "example" in the file name and that end with ".gz" from the given URL
    url = "https://example.com/files"
    instrument_glob_pattern = ["example"]
    substring_must_match = ".gz"
    content = extract_content(url, instrument_glob_pattern, substring_must_match)
    print(content)
    # Output: {'file_name': ['example1.gz', 'example2.gz'], 'date': ['2022-01-01', '2022-02-01'], 'size': ['1000', '2000'], 'date_changed': ['2022-01-01', '2022-02-01']}
    """

    LOGGER.info(
        f"Extracting files with the following substrings: {substring_must_match}"
    )
    soup = fetch_content(url)
    content = {"file_name": [], "date": [], "size": [], "date_changed": []}
    # If the substrings_to_include is not a list, make it a list if it is not None
    if substring_must_match is None:
        substring_must_match = ".fit.gz"

    for link in soup.find_all("a"):
        if substring_must_match in link.get("href"):
            if instrument_glob_pattern is None or 
                content["file_name"].append(link.get("href"))
                if return_date_size:
                    date_changed, size = extract_date_size_from_soup(
                        soup, content["file_name"][-1]
                    )
                    content["date_changed"].append(date_changed)
                    content["size"].append(size)
    LOGGER.info(
        f"Extracted {len(content)} files with the following substrings: {substrings_to_include} and the following substring must match: {substring_must_match}"
    )
    if len(content) > 0:
        LOGGER.info(f"Example of extracted files: {content['file_name'][:2]}")
    else:
        LOGGER.info(f"No files extracted")
    return content


def extract_fiz_gz_files_urls(
    year, month, day, instrument_glob_pattern, return_date_size=True
):
    """
    Extracts all the .fit.gz files from the given year, month and day
    instrument_substring: If specified, only files with the instrument_substring name will be extracted

    Returns a list of all the .fit.gz files
    """
    url = f"{FILES_BASE_URL}{year}/{month}/{day}/"
    content = extract_content(
        url,
        instrument_glob_pattern=instrument_glob_pattern,
        return_date_size=return_date_size,
    )
    # Add the base url to the file names
    urls = [url + file_name for file_name in content["file_name"]]
    content["url"] = urls
    # Extract the date from the file name
    content["date"] = [extract_date_from_path(url) for url in urls]
    LOGGER.info(f"Extracted {len(urls)} files")
    return content


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


def get_urls(start_date, end_date, instrument_glob_pattern) -> list[str]:
    """
    Get the urls of fiz gz files for a given date range and instrument_glob_pattern.

    Parameters
    ----------
    start_date : pd.Datetime
        The start date.
    end_date : pd.Datetime
        The end date.
    instrument_glob_pattern : None or list of str
        The instrument_glob_pattern name. If None, all instruments are considered. Also, the capitalization is ignored.

    Returns
    -------
    list of str
        The list of urls of fiz gz files.
    """

    content = {"file_name": [], "url": [], "date": [], "size": [], "date_changed": []}
    for date in tqdm(pd.date_range(start_date, end_date), desc="fetching urls"):
        content_ = extract_fiz_gz_files_urls(
            date.year,
            str(date.month).zfill(2),
            str(date.day).zfill(2),
            instrument_glob_pattern=instrument_glob_pattern,
        )
        LOGGER.debug(f"extracted {len(content_)} files for {date}")
        for key in content:
            content[key].extend(content_[key])

    return content


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
    content = get_urls(start_date, end_date, instrument)
    urls = content["url"]
    # Create a partial function to pass the dir argument and return_download_path
    fn = partial(
        download_ecallisto_file, return_download_path=return_download_paths, dir=dir
    )
    # Multiprocessing via tqdm
    with Pool() as p:
        r = list(tqdm(p.imap(fn, urls), total=len(urls), desc="Downloading files"))

    if return_download_paths:
        return r


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
        previous_status,
        how="outer",
        indicator=True,
        on=["file_name", "url", "date_changed", "date", "size"],
        suffixes=("", "_prev"),
    )
    # Keep only the rows which are only in the current status.
    diff = diff[diff["_merge"] == "left_only"]
    diff = diff.drop(columns=["_merge"])
    return diff


if __name__ == "__main__":
    print(f"Downloading files to {LOCAL_DATA_FOLDER}")
    download_ecallisto_files(
        dir=LOCAL_DATA_FOLDER,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 2),
        instrument="ALASKA",
    )
