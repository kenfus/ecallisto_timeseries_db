import asyncio
import io
import json
import os
import uuid
from typing import List, Optional

from astropy.io import fits
from astropy.table import Table
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from database_functions import (
    sql_result_to_df,
    timebucket_values_from_database_sql,
    values_from_database_sql,
)
from database_utils import get_table_names_sql

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
    version="0.1",
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
    return_type: str = Field(
        "json", description="The desired return type", enum=["json", "fits"]
    )
    columns: Optional[List[str]] = Field(
        None, description="List of columns to include in the response", example=None
    )


# Add the BackgroundTasks parameter to your function
@app.post("/api/data")
async def get_data(background_tasks: BackgroundTasks, data_request: DataRequest):
    data_request_dict = data_request.dict()
    data_request_dict["table"] = data_request_dict.pop("instrument_name")

    # Create a unique ID for this request and generate a filename based on this
    file_id = str(uuid.uuid4())
    file_path_json = f"data/{file_id}.json"
    file_path_fits = f"data/{file_id}.fits"

    # Add a task to run in the background that will get the data and save it to a file
    background_tasks.add_task(get_and_save_data, data_request_dict, file_path_json, file_path_fits)

    # Return the URLs where the files will be available once the data has been fetched
    return {"json_url": f"/api/data/{file_id}.json", "fits_url": f"/api/data/{file_id}.fits"}

async def get_and_save_data(data_request_dict, file_path_json, file_path_fits):
    if not any([data_request_dict["timebucket"], data_request_dict["agg_function"]]):
        data = values_from_database_sql(**data_request_dict)
    else:
        data = timebucket_values_from_database_sql(**data_request_dict)

    # Create dir 
    os.makedirs(os.path.dirname(file_path_json), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_fits), exist_ok=True)

    # Change data to dataframe
    df = sql_result_to_df(data)

    # Save as JSON
    df.to_json(file_path_json)

    # Save as FITS
    table = Table.from_pandas(df.dropna(axis=1, how="all"))
    hdu = fits.table_to_hdu(table)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(file_path_fits)


@app.get("/api/data/{file_id}.json")
async def get_json(file_id: str):
    file_path = f"data/{file_id}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        return {"error": "File not found. The data may still be fetching, or the file ID may be incorrect."}


@app.get("/api/data/{file_id}.fits")
async def get_fits(file_id: str):
    file_path = f"data/{file_id}.fits"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return StreamingResponse(file, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{file_id}.fits"'})
    else:
        return {"error": "File not found. The data may still be fetching, or the file ID may be incorrect."}