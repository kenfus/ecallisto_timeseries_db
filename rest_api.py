import io
from typing import List

from astropy.io import fits
from astropy.table import Table
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from database_functions import timebucket_values_from_database_sql
from database_utils import get_table_names_sql, sql_result_to_df

"""
Start the REST API with:
uvicorn rest_api:app --reload (still in development, after it should be started as follows: gunicorn -k uvicorn.workers.UvicornWorker rest_api:app )
You can then access the API at 
http://127.0.0.1:8000/docs
(with port forwading.)
or http://147.86.10.169/docs 
when inside the FHNW network.
"""

app = FastAPI(
    title="E-Callisto REST API",
    description="REST API for the E-Callisto database",
    version="0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

class DataRequest(BaseModel):
    instrument_name: str = Field('austria_unigraz_01', description='The name of the instrument', enum=get_table_names_sql())
    start_datetime: str = Field('2021-03-10 06:30:00', description='The start datetime for the data request')
    end_datetime: str = Field('2021-03-14 23:30:00', description='The end datetime for the data request')
    timebucket: str = Field('5m', description='The time bucket for aggregation')
    agg_function: str = Field('MAX', description='The aggregation function')
    return_type: str = Field('json', description='The desired return type')
    columns: List[str] = Field(None, description='List of columns to include in the response')

@app.get("/")
def root():
    return RedirectResponse(url="/redoc")

@app.post("/data")
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
    data_request_dict['table'] = data_request_dict.pop('instrument_name')

    # Get data
    data = timebucket_values_from_database_sql(**data_request_dict)

    # Change data to dataframe
    df = sql_result_to_df(data)

    if data_request_dict['return_type'] == 'json':
        return df.to_dict()
    
    elif data_request_dict['return_type'] == 'fits':
        # Write the table to a FITS file in memory
        table = Table.from_pandas(df)
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
        filename = f"{data_request_dict['instrument_name']}_{data_request_dict['start_datetime']}_{data_request_dict['end_datetime']}.fits"

        # Return the FITS file as a downloadable attachment
        return StreamingResponse(
            iterfile(),
            media_type='application/octet-stream',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

