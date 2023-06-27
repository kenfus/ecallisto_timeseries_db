import io
from typing import List, Optional

from astropy.io import fits
from astropy.table import Table
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from database_functions import (timebucket_values_from_database_sql,
                                values_from_database_sql, sql_result_to_df)
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


@app.get("/api/")
def root():
    return RedirectResponse(url="/api/redoc")


@app.post("/api/data")
def get_data(data_request: DataRequest):
    """
    Retrieve data based on the provided data request.

    This route retrieves data from the E-Callisto database based on the specified parameters in the data request.

    - `instrument_name`: The name of the instrument.
    - `start_datetime`: The start datetime for the data request.
    - `end_datetime`: The end datetime for the data request.
    - `timebucket`: The time bucket for aggregation.
    - `agg_function`: The aggregation function.
    - `return_type`: The desired return type.
    - `columns`: List of columns to include in the response.

    Returns:
        - If `return_type` is 'json': The data as a JSON object.
        - If `return_type` is 'fits': The data as a downloadable FITS file.

    """
    data_request_dict = data_request.dict()
    data_request_dict["table"] = data_request_dict.pop("instrument_name")

    # Get data
    if not any([data_request_dict["timebucket"], data_request_dict["agg_function"]]):
        data = values_from_database_sql(**data_request_dict)
    else:
        data = timebucket_values_from_database_sql(**data_request_dict)

    # Change data to dataframe
    df = sql_result_to_df(data)

    if data_request_dict["return_type"] == "json":
        return df.to_dict()

    elif data_request_dict["return_type"] == "fits":
        return return_fits(df, data_request_dict)


def return_fits(df, data_request_dict):
    # Write the table to a FITS file in memory
    table = Table.from_pandas(df.dropna(axis=1, how="all"))
    file_like = io.BytesIO()
    hdu = fits.table_to_hdu(table)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(file_like)

    # Move the cursor to the start of the BytesIO object
    file_like.seek(0)

    # Define a generator that yields file data in chunks
    def iterfile():
        chunk_size = 8192
        while True:
            chunk = file_like.read(chunk_size)
            if not chunk:
                break
            yield chunk

    # Define the filename for the download
    filename = f"{data_request_dict['table']}_{data_request_dict['start_datetime']}_{data_request_dict['end_datetime']}.fits"

    # Return the FITS file as a downloadable attachment
    return StreamingResponse(
        iterfile(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
