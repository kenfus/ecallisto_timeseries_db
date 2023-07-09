import os
import time
from datetime import datetime

import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from bulk_load_to_database_between_dates import (
    add_instruments_from_paths_to_database, add_specs_from_paths_to_database,
    create_dict_of_instrument_paths)
from data_creation_utils import FILES_LOCAL_PATH
from logging_utils import setup_custom_logger

LOGGER = setup_custom_logger("add_new_files_to_db")


def add_data_to_database(path, replace):
    LOGGER.info(f"Adding data from {path} to database.")
    # To list
    path = [path]
    # Check if there are new instruments
    dict_paths = create_dict_of_instrument_paths(path)
    # Add the instruments to the database
    add_instruments_from_paths_to_database(dict_paths)
    # Add the dat a to the database
    add_specs_from_paths_to_database(path, 1, 1, replace=replace)


def create_observer_for_day(date, event_handler):
    path = os.path.join(
        FILES_LOCAL_PATH,
        str(date.year),
        str(date.month).zfill(2),
        str(date.day).zfill(2),
    )
    if not os.path.exists(path):
        return None
    LOGGER.info(f"Creating observer for {path}.")
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    return observer


def get_file_path(event):
    if event.is_directory:
        return None  # Ignore directory modifications

    file_path = event.src_path
    if event.event_type == "moved":
        file_path = event.dest_path  # Use the destination path if it was a move event

    return file_path


class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        print(event)
    def on_modified(self, event):
        path = get_file_path(event)
        if path and path.endswith(".fit.gz"):
            LOGGER.info(f"Detected modifcation of {path}.")
            add_data_to_database(path, replace=True)

    def on_moved(self, event):
        path = get_file_path(event)
        if path and path.endswith(".fit.gz"):
            LOGGER.info(f"Detected creation of {path}.")
            add_data_to_database(path, replace=False)


def observe_days(days_to_observe):
    ###
    date_range = pd.date_range(
        pd.Timestamp.now() - pd.DateOffset(days=days_to_observe),
        pd.Timestamp.now(),
        freq="D",
    )
    LOGGER.info(
        f"Checking for new data in the last {days_to_observe} days. First day: {date_range[0].date()}, last day: {date_range[-1].date()}."
    )
    LOGGER.info(
        f"Will start observing {len(date_range)} days and thus starting {len(date_range)} observers."
    )
    # Observers
    observers = []
    event_handler = Handler()
    # Create current year and months to check
    current_date = datetime.now().date()
    try:
        for date in date_range:
            observer = create_observer_for_day(date, event_handler)
            observers.append(observer)

        while True:
            time.sleep(1)
            now = datetime.now()
            if (
                now.month > current_date.month
                or now.year > current_date.year
                or now.day > current_date.day
            ):
                # Create new observer
                observer = create_observer_for_day(now, event_handler)
                if observer == None:
                    continue
                observers.append(observer)
                LOGGER.info(
                    f"New day has started. Stopping observers and creating new observers for {now.year}-{now.month}."
                )
                LOGGER.info(
                    f"Created new observer for {now}. There are now {len(observers)} observers running."
                )
                # Stop the observers
                observers.pop(0).stop()
                # New day has started
                current_date = now.date()

    except KeyboardInterrupt:
        for observer in observers:
            observer.stop()

    for observer in observers:
        observer.join()


if __name__ == "__main__":
    DAYS_TO_CHECK = 1
    observe_days(DAYS_TO_CHECK)
