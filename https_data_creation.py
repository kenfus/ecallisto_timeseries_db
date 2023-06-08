import datetime
import logging
import re
import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing.pool import Pool as Pool
from typing import Union

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from data_creation import LOCAL_DATA_FOLDER, MIN_FILE_SIZE, extract_date_from_path

LOGGER = logging.getLogger("database_data_addition")
FILES_BASE_URL = "http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"

# Requests session
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

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

def fetch_content(url):
    reqs = session.get(url)
    soup = BeautifulSoup(reqs.text, "lxml")
    LOGGER.info(f"Fetching content from {url}")
    LOGGER.info(f"Status code: {reqs.status_code}")
    return soup


def extract_fiz_gz_files_urls(
    year, month, day, instrument_regexr_pattern, return_date_size=False
):
    """
    Extracts all the .fit.gz files from the given year, month and day
    instrument_substring: If specified, only files with the instrument_substring name will be extracted

    Returns a list of all the .fit.gz files
    """
    url = f"{FILES_BASE_URL}{year}/{month}/{day}/"
    content = extract_content(
        url,
        instrument_regexr_pattern=instrument_regexr_pattern,
        return_date_size=return_date_size,
    )
    # Add the base url to the file names
    urls = [url + file_name for file_name in content["file_name"]]
    content["path"] = urls
    # Extract the date from the file name
    content["date"] = [extract_date_from_path(url) for url in urls]
    LOGGER.info(f"Extracted {len(urls)} files")
    return content

def extract_content(
    url,
    instrument_regexr_pattern,
    substring_must_match="fit.gz",
    return_date_size=False,
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
    return_date_size: bool
        If the file size and date of creation should be added. Currently very slow.

    Returns
    -------
    list of str
    The list of URLs of .fiz.gz files.

    Example:
    # Extract all files that include "example" in the file name and that end with ".gz" from the given URL
    url = "https://example.com/files"
    instrument_regexr_pattern = "example"
    substring_must_match = ".gz"
    content = extract_content(url, instrument_regexr_pattern, substring_must_match)
    print(content)
    # Output: {'file_name': ['example1.gz', 'example2.gz'], 'date': ['2022-01-01', '2022-02-01'], 'size': ['1000', '2000'], 'date_changed': ['2022-01-01', '2022-02-01']}
    """

    LOGGER.info(
        f"Extracting files with the following must-match substrings: {substring_must_match}"
    )
    soup = fetch_content(url)
    content = {"file_name": [], "date": [], "size": [], "date_changed": []}
    # If the substrings_to_include is not a list, make it a list if it is not None
    if substring_must_match is None:
        substring_must_match = ".fit.gz"

    for link in soup.find_all("a"):
        href = link.get("href")
        if substring_must_match in href:
            regex_to_match = re.compile(instrument_regexr_pattern, re.IGNORECASE) if instrument_regexr_pattern is not None else None
            if instrument_regexr_pattern is None or re.search(regex_to_match, href):
                content["file_name"].append(href)
                if False: #return_date_size: Currently INCREDIBLE SLOW
                    date_changed, size = extract_date_size_from_soup(
                        soup, content["file_name"][-1]
                    )
                    content["date_changed"].append(date_changed)
                    content["size"].append(size)
    LOGGER.info(
        f"Extracted {len(content)} files with the following regexr pattern: {instrument_regexr_pattern} and the following substring must match: {substring_must_match}"
    )
    if len(content) > 0:
        LOGGER.info(f"Example of extracted files: {content['file_name'][:2]}")
    else:
        LOGGER.info(f"No files extracted")
    return content


def extract_fiz_gz_files_urls(
    year, month, day, instrument_regexr_pattern, return_date_size=False
):
    """
    Extracts all the .fit.gz files from the given year, month and day
    instrument_substring: If specified, only files with the instrument_substring name will be extracted

    Returns a list of all the .fit.gz files
    """
    url = f"{FILES_BASE_URL}{year}/{month}/{day}/"
    content = extract_content(
        url,
        instrument_regexr_pattern=instrument_regexr_pattern,
        return_date_size=return_date_size,
    )
    # Add the base url to the file names
    urls = [url + file_name for file_name in content["file_name"]]
    content["path"] = urls
    # Extract the date from the file name
    content["date"] = [extract_date_from_path(url) for url in urls]
    LOGGER.info(f"Extracted {len(urls)} files")
    return content



def get_urls(start_date, end_date, instrument_regexr_pattern = None) -> list[str]:
    """
    Get the urls of fiz gz files for a given date range and instrument_glob_pattern.

    Parameters
    ----------
    start_date : pd.Datetime
        The start date.
    end_date : pd.Datetime
        The end date.
    instrument_regexr_pattern : None or list of str
        The instrument_regexr_pattern name. If None, all instruments are considered. Also, the capitalization is ignored.

    Returns
    -------
    list of str
        The list of urls of fiz gz files.
    """
    content = {"file_name": [], "path": [], "date": [], "size": [], "date_changed": []}
    for date in tqdm(pd.date_range(start_date, end_date, inclusive='both'), desc="fetching urls"):
        content_ = extract_fiz_gz_files_urls(
            date.year,
            str(date.month).zfill(2),
            str(date.day).zfill(2),
            instrument_regexr_pattern=instrument_regexr_pattern,
        )
        LOGGER.debug(f"extracted {len(content_)} files for {date}")
        for key in content:
            content[key].extend(content_[key])

    return content



def download_ecallisto_files(
    dir,
    start_date=datetime.today().date() - timedelta(days=1),
    end_date=datetime.today().date(),
    instruments: Union[list, None] = None,
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
    instruments : list, optional
        If specified, only files containing any of the given instruments (regexr pattern) will be downloaded. Default is None (all instruments).
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
    if isinstance(instruments, str) and instruments.lower() in ["*", "all"]:
        instruments = None
    LOGGER.info(
        f"Downloading files from {start_date} to {end_date} (instrument: {instruments if instruments else 'all'})"
    )
    if isinstance(instruments, str):
        instruments = [instruments]
    urls = []
    for instrument in instruments:
        instrument_regexr_pattern = instrument_name_to_regex_pattern(instrument)
        content = get_urls(start_date, end_date, instrument_regexr_pattern)
        url = content["path"]
        urls.extend(url)
    # Create a partial function to pass the dir argument and return_download_path
    fn = partial(
        download_ecallisto_file, return_download_path=return_download_paths, dir=dir
    )
    # Multiprocessing via tqdm
    with Pool() as p:
        r = list(tqdm(p.imap(fn, urls), total=len(urls), desc="Downloading files"))

    if return_download_paths:
        return r

def instrument_name_to_regex_pattern(instrument_name):
    """
    Convert an instrument name to a regex pattern.

    Parameters
    ----------
    instrument_name : str
        The instrument name.

    Returns
    -------
    str
        The regex pattern.

    Examples
    --------
    >>> instrument_name_to_regex_pattern("ALASKA-COHOE-62")
    '([ALASKA\-COHOE]+_\d{8}_\d{6}_62.+)'
    >>> instrument_name_to_regex_pattern("ALASKA-COHOE")
    '([ALASKA\-COHOE]+_\d{8}_\d{6}.+)'
    >>> instrument_name_to_regex_pattern("ALASKA")
    '([ALASKA]+_\d{8}_\d{6}.+)'
    >>> instrument_name_to_regex_pattern(None)
    '([A-Za-z0-9\-]+_\d{8}_\d{6}.+)'
    """
    if instrument_name is None:
        return "([A-Za-z0-9\-]+_\d{8}_\d{6}.+)"
    pattern = "_\d{8}_\d{6}"  # date and time
    if instrument_name[-2:].isdigit():
        pattern = pattern + "_" + instrument_name[-2:] + ".+"  # instrument number
        name = instrument_name[:-3]
    else:
        pattern = pattern + ".+"
        name = instrument_name
    pattern = "(" + name.replace("-", "\-") + pattern + ")"
    return pattern