import json
import os
import time
import hashlib
from typing import List, Optional
from datetime import datetime, timedelta
from astropy.io import fits
from astropy.table import Table
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from database_functions import (
    sql_result_to_df,
    timebucket_values_from_database_sql,
    values_from_database_sql,
    get_table_names_with_data_between_dates_sql,
)
from database_utils import get_table_names_sql
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
    version="0.2",
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
async def get_data(background_tasks: BackgroundTasks, data_request: DataRequest):
    data_request_dict = data_request.dict()
    data_request_dict["table"] = data_request_dict.pop("instrument_name")

    LOGGER.info(f"Received data request: {data_request_dict} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create a unique ID for this request and generate a filename based on this
    file_id = get_sha256_from_dict(data_request_dict)
    metadata_path_json = f"data/{file_id}.json"
    file_path_parquet = f"data/{file_id}.parquet"

    # Check if the file already exists
    if os.path.exists(metadata_path_json) and os.path.exists(file_path_parquet):
        LOGGER.info(f"Data request: {data_request_dict} already exists at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        LOGGER.info(f"Data request: {data_request_dict} is new at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Add a task to run in the background that will get the data and save it to a file
        background_tasks.add_task(get_and_save_data, data_request_dict, file_path_parquet, metadata_path_json)

    # Return the URLs where the files will be available once the data has been fetched
    return {"data_parquet_url": f"/api/data/{file_id}.parquet", "info_json_url": f"/api/data/{file_id}.json", "file_id": file_id}

async def get_and_save_data(data_request_dict, file_path_parquet, info_json_url):
    try:
        if not any([data_request_dict["timebucket"], data_request_dict["agg_function"]]):
            data = values_from_database_sql(**data_request_dict)
        else:
            data = timebucket_values_from_database_sql(**data_request_dict)


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

    # Save as JSON
    df.to_parquet(file_path_parquet, compression='gzip')

@app.get("/api/tables")
async def get_tables():
    table_names = get_table_names_sql()
    LOGGER.info(f"Delivering table names at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    table_names = [table_name for table_name in table_names if table_name not in ["test", "test2"]]
    return {"tables": table_names}

class DataAvailabilityRequest(BaseModel):
    end_datetime: str = Field(
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), description="The end datetime for the data availability check"
    )
    start_datetime: str = Field(
        (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), description="The start datetime for the data availability check"
    )


@app.post("/api/data_availability")
async def get_table_names_with_data_between_dates(request: DataAvailabilityRequest):
    LOGGER.info(f"Checking table data availability for: {request.dict()} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    table_names_with_data = get_table_names_with_data_between_dates_sql(request.start_datetime, request.end_datetime)
    # Remove test table etc.
    table_names_with_data = [table_name for table_name in table_names_with_data if table_name not in ["test", "test2"]]
    return {"table_names": table_names_with_data}

@app.get("/api/data/{file_id}.json")
async def get_json(file_id: str):
    file_path = f"data/{file_id}.json"
    LOGGER.info(f"Delivering data request: {file_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        return {"error": "File not found. The data may still be fetching, or the file ID may be incorrect."}
    
@app.get("/api/data/{file_id}.parquet")
async def get_parquet(file_id: str):
    file_path = f"data/{file_id}.parquet"
    LOGGER.info(f"Delivering data request: {file_id} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if os.path.exists(file_path):
        return StreamingResponse(open(file_path, "rb"), media_type="application/octet-stream")
    else:
        return {"error": "File not found. The data may still be fetching, or the file ID may be incorrect."}
    
def get_sha256_from_dict(data: dict) -> str:
    """Generates a SHA256 hash from a dictionary."""
    data_string = json.dumps(data, sort_keys=True)  # we use sort_keys to ensure consistent ordering
    return hashlib.sha256(data_string.encode()).hexdigest()