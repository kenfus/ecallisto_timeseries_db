import asyncio
import json
import os
import time
import hashlib
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from database_functions import (
    sql_result_to_df,
    timebucket_values_from_database_sql,
    values_from_database_sql,
    get_table_names_with_data_between_dates_sql,
    check_if_table_has_data_between_dates_sql,
    get_min_max_datetime_from_table_sql,
    get_column_names_sql,
    timebucket_to_seconds
)
## To add meta data
from database_utils import get_table_names_sql, get_last_spectrogram_from_paths_list, instrument_name_to_glob_pattern
from bulk_load_to_database_between_dates import get_paths # TODO: Move this get_paths to another utils file.
##
import logging_utils
LOGGER = logging_utils.setup_custom_logger("rest_api")

"""
Start the REST API with:
uvicorn rest_api:app --reload (still in development, after it should be started as follows: gunicorn -k uvicorn.workers.UvicornWorker rest_api:app )
You can then access the API at 
http://127.0.0.1:8000/docs
(with port forwading.)
or https://v000792.fhnw.ch/api/
"""

app = FastAPI(
    title="E-Callisto REST API",
    openapi_url="/api/openapi.json",
    description="REST API for the E-Callisto database",
    version="0.3",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


class DataRequest(BaseModel):
    instrument_name: str = Field(
        "austria_unigraz_01",
        description="The name of the instrument",
        enum=get_table_names_sql(),
        example="austria_unigraz_01",
    )
    start_datetime: str = Field(
        "2021-03-10 06:30:00", description="The start datetime for the data request"
    )
    end_datetime: str = Field(
        "2021-03-14 23:30:00", description="The end datetime for the data request"
    )
    timebucket: str = Field(
        None, description="The time bucket for aggregation", example="1h"
    )
    agg_function: str = Field(
        None,
        description="The aggregation function",
        enum=["MIN", "MAX", "AVG", "MEDIAN"],
        example="MAX",
    )
    columns: Optional[List[str]] = Field(
        None, description="List of columns to include in the response", example=None
    )

# Add the BackgroundTasks parameter to your function
@app.post("/api/data")
def get_data(background_tasks: BackgroundTasks, data_request: DataRequest):
    data_request_dict = data_request.dict()
    data_request_dict["table"] = data_request_dict.pop("instrument_name")

    LOGGER.info(f"Received data request: {data_request_dict} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create a unique ID for this request and generate a filename based on this
    file_id = get_sha256_from_dict(data_request_dict)
    info_json_url = f"data/{file_id}.json"
    file_path_parquet = f"data/{file_id}.parquet"
    meta_data_url = f"data/{file_id}_meta_data.json"
    
    # Create json with information that we are processing the request
    with open(info_json_url, "w") as f:
        json.dump({"status": "processing"}, f)

    # Check if the file already exists
    if os.path.exists(info_json_url) and os.path.exists(file_path_parquet):
        LOGGER.info(f"Data request: {data_request_dict} already exists at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        LOGGER.info(f"Data request: {data_request_dict} is new at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Add a task to run in the background that will get the data and save it to a file
        background_tasks.add_task(get_and_save_data, data_request_dict, file_path_parquet, info_json_url, meta_data_url)

    # Return the URLs where the files will be available once the data has been fetched
    return {"data_parquet_url": f"/api/data/{file_id}.parquet", "info_json_url": f"/api/data/{file_id}.json", "file_id": file_id, 'meta_data_url': f"/api/data/{file_id}_meta_data.json"}

def get_and_save_data(data_request_dict, file_path_parquet, info_json_url, meta_data_url):
    size_of_request_mb = calculate_size_of_request(data_request_dict)
    LOGGER.info(f"Size of request: {size_of_request_mb} MB")
    
    try:
        if size_of_request_mb > 50:
            raise ValueError(f"Request too large. Size of request: {size_of_request_mb} MB. Max size is 50MB.")
        if data_request_dict["timebucket"] and data_request_dict["agg_function"]:
            data = timebucket_values_from_database_sql(**data_request_dict)
        else:
            data = values_from_database_sql(**data_request_dict)

    except ValueError as e:
        LOGGER.error(f"Error in data request: {data_request_dict} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        LOGGER.error(e)
        # Write error message into file
        with open(info_json_url, "w") as f:
            json.dump({"error": str(e)}, f)
        return
    
    if len(data) == 0:
        LOGGER.error(f"Error in data request: {data_request_dict} at {time.strftime('%Y-%m-%d %H:%M:%S')}. No data found.")
        LOGGER.error("No data found")
        with open(info_json_url, "w") as f:
            json.dump({"error": "No data found. Check your request?"}, f)
        return
    
    # Logg
    LOGGER.info(f"Finished data request: {data_request_dict} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create dir 
    os.makedirs(os.path.dirname(info_json_url), exist_ok=True)

    # Change data to dataframe
    df = sql_result_to_df(data)

    ## Add metadata
    try:
        meta_data = return_header_from_newest_spectogram(df, data_request_dict["table"])
    except Exception as e:
        meta_data = {}
        meta_data["error"] = str(e)

    # Save as DF
    ## TODO: Replace parquet with something that supports meta data?
    df.to_parquet(file_path_parquet, compression='gzip')
    del df

    # Save json with metadata
    with open(meta_data_url, "w") as f:
        json.dump(meta_data, f)

    # Save json all ok
    with open(info_json_url, "w") as f:
        json.dump({"status": "ok"}, f)

@app.get("/api/tables")
def get_tables():
    table_names = get_table_names_sql()
    LOGGER.info(f"Delivering table names at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    table_names = [table_name for table_name in table_names if table_name not in ["test", "test2"]]
    return {"tables": table_names}

class TableDataCheckRequest(BaseModel):
    instrument_name: str = Field(
        ...,
        description="The name of the table to check",
        enum=get_table_names_sql(),
        example="austria_unigraz_01",
    )
    start_datetime: str = Field(
        ...,
        description="The start datetime for the data check"
    )
    end_datetime: str = Field(
        ...,
        description="The end datetime for the data check"
    )

@app.post("/api/table_data_check")
def check_table_data_availability(request: TableDataCheckRequest):
    LOGGER.info(f"Checking data availability for table: {request.instrument_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    has_data = check_if_table_has_data_between_dates_sql(request.instrument_name, request.start_datetime, request.end_datetime)
    LOGGER.info(f"Table: {request.instrument_name} has data: {has_data}")
    return {"instrument_name": request.instrument_name, "has_data": has_data}


class DataAvailabilityRequest(BaseModel):
    end_datetime: str = Field(
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description="The end datetime for the data availability check"
    )
    start_datetime: str = Field(
        (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), description="The start datetime for the data availability check"
    )

@app.post("/api/data_availability")
def get_table_names_with_data_between_dates(request: DataAvailabilityRequest):
    LOGGER.info(f"Checking table data availability for: {request.dict()} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    table_names_with_data = get_table_names_with_data_between_dates_sql(request.start_datetime, request.end_datetime)
    # Remove test table etc.
    table_names_with_data = [table_name for table_name in table_names_with_data if table_name not in ["test", "test2"]]
    return {"table_names": table_names_with_data}

class TableNameRequest(BaseModel):
    instrument_name: str

@app.post("/api/min_max_datetime")
def get_min_max_datetime(request: TableNameRequest):
    LOGGER.info(f"Fetching min and max datetime from: {request.instrument_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        min_max_datetime = get_min_max_datetime_from_table_sql(request.instrument_name)
        return {"min_datetime": min_max_datetime[0], "max_datetime": min_max_datetime[1]}
    except Exception as e:
        LOGGER.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=400, detail="Error in fetching data.")


@app.get("/api/data/{file_id}.json")
def get_json(file_id: str):
    file_path = f"data/{file_id}.json"
    if os.path.exists(file_path):
        LOGGER.info(f"Delivering data request: {file_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        LOGGER.info(f"Data request: {file_id} not found at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        raise HTTPException(status_code=204, detail="File not found.")

@app.get("/api/data/{file_id}_meta_data.json")
def get_json(file_id: str):
    file_path = f"data/{file_id}_meta_data.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            LOGGER.info(f"Delivering data request: {file_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return json.load(file)
    else:
        LOGGER.info(f"Data request: {file_id} not found at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        raise HTTPException(status_code=204, detail="File not found.")
    
@app.get("/api/data/{file_id}.parquet")
def get_parquet(file_id: str):
    file_path = f"data/{file_id}.parquet"
    if os.path.exists(file_path):
        LOGGER.info(f"Delivering data request: {file_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return StreamingResponse(open(file_path, "rb"), media_type="application/octet-stream")
    else:
        LOGGER.info(f"Data request: {file_id} not found at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        raise HTTPException(status_code=204, detail="File not found. Check the JSON for errors.")
    
def get_sha256_from_dict(data: dict) -> str:
    """Generates a SHA256 hash from a dictionary."""
    data_string = json.dumps(data, sort_keys=True)  # we use sort_keys to ensure consistent ordering
    return hashlib.sha256(data_string.encode()).hexdigest()

def calculate_size_of_request(data_request_dict):
    """Calculate the size of the request in MB."""
    # Calculate the number of columns
    col_num = len(get_column_names_sql(data_request_dict["table"]))
    # Calculate the number of seconds in the request
    start_datetime = datetime.strptime(data_request_dict["start_datetime"], "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(data_request_dict["end_datetime"], "%Y-%m-%d %H:%M:%S")
    # Calculate the number of seconds in the request
    time_delta_seconds = (end_datetime - start_datetime).total_seconds()
    # Calculate the number of rows
    row_num = time_delta_seconds / timebucket_to_seconds(data_request_dict["timebucket"])
    # Calculate the size of the request in MB
    return (col_num * row_num * 8) / 1024 / 1024


async def remove_old_files():
    while True:
        now = datetime.now()
        n_files = len(os.listdir("data"))
        for f in os.listdir("data"):
            f = os.path.join("data", f)
            if os.path.isfile(f):
                timestamp = os.path.getmtime(f)
                file_time = datetime.fromtimestamp(timestamp)
                if now - file_time > timedelta(hours=24):
                    os.remove(f)
        LOGGER.info(f"Removed {n_files - len(os.listdir('data'))} files.")
        await asyncio.sleep(60 * 60 * 24)  # sleep for 24 hours


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(remove_old_files())
    
### Other helping functions
def return_header_from_newest_spectogram(df, instrument_name):
    """
    Add the header from the newest spectrogram (based on the datetime inside the df)
    to the dataframe.
    """
    df = df.copy()
    # Get last day from df
    last_day = df.index.max().date()
    # Get glob pattern
    glob_pattern = instrument_name_to_glob_pattern(instrument_name)
    # Get paths
    paths = get_paths(last_day, last_day, glob_pattern)
    # Get last spectrogram
    last_spectrogram = get_last_spectrogram_from_paths_list(paths)
    dict_ = {}
    # Add metadata
    for key, value in last_spectrogram.header.items():
        dict_[key] = value

    del last_spectrogram
    return dict_