from fastapi import FastAPI
from typing import List
from database_functions import timebucket_values_from_database_sql

app = FastAPI()

@app.get("/data")
def get_data(
    instrument_name: str,
    start_datetime: str,
    end_datetime: str,
    timebucket: str = '1m',
    agg_function: str = 'MAX',
    columns: List[str] = None,
    quantile_value: float = None,
    columns_not_to_select: List[str] = ["datetime", "burst_type"],
):
    return timebucket_values_from_database_sql(
        instrument_name,
        start_datetime,
        end_datetime,
        columns,
        timebucket,
        agg_function,
        quantile_value,
        columns_not_to_select,
    )