from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from database_functions import timebucket_values_from_database_sql
from database_utils import sql_result_to_df
from astropy.table import Table
from astropy.io import fits
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

class DataRequest(BaseModel):
    instrument_name: str = 'austria_unigraz_01'
    start_datetime: str = '2021-03-10 21:30:00'
    end_datetime: str = '2021-03-11 21:30:00'
    timebucket: str = '1m'
    agg_function: str = 'MAX'
    columns: List[str] = None
    quantile_value: float = None
    columns_not_to_select: List[str] = ["datetime", "burst_type"]

@app.post("/data")
def get_data(data_request: DataRequest):
    data_request_dict = data_request.dict()
    data_request_dict['table'] = data_request_dict.pop('instrument_name')

    # Get data
    data = timebucket_values_from_database_sql(**data_request_dict)

    # Change data to dataframe
    df = sql_result_to_df(data)

    # Write the table to a FITS file in memory
    table = Table.from_pandas(df)
    file_like = io.BytesIO()
    hdu = fits.table_to_hdu(table)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(file_like)

    # Define the filename for the download
    filename = 'data.fits'

    # Return the FITS file as a downloadable attachment
    return StreamingResponse(
        file_like.getvalue(),
        media_type='application/octet-stream',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )



