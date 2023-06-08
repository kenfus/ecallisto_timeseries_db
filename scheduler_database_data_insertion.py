import time
from datetime import datetime, timedelta
from subprocess import PIPE, STDOUT, Popen

import schedule

from logging_utils import setup_custom_logger

LOGGER = setup_custom_logger("scheduler_database_data_addition")


# Schedule the script to run every 5 seconds
def job():
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    process = Popen(
        [
            "python",
            "continiously_add_all_data_to_database.py",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
        ],
        stderr=STDOUT,
        stdout=PIPE,
    )
    LOGGER.info(
        f"Running script to add data to the database from {start_date} to {end_date}."
    )
    # Wait for process completion and capture output
    stdout, _ = process.communicate()

    # Log errors
    if process.returncode != 0:
        LOGGER.error(
            f"Error running script to add data to the database from {start_date} to {end_date}."
        )
        LOGGER.error(f"Script output: {stdout.decode()}")


# Run the job immediately
job()

# Schedule the job to run every 5 seconds
schedule.every(60).minutes.do(job)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
