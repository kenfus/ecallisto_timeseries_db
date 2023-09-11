import os
import re
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import psycopg2
except ImportError:
    print("psycopg2 not installed, DB functions will not work (which can be ok).")
from dateutil.parser import parse as dateutil_parse
from pytimeparse import parse as pytimeparse_parse
from logging_utils import GLOBAL_LOGGER as LOGGER

# Create variables for the connection to the OS
os.environ["PGHOST"] = "localhost"
# If no user is set, set it to postgres because that is the default user and it's hopefully not production
if "PGUSER" not in os.environ:
    os.environ["PGUSER"] = "ecallisto_read_only"
# If no password is set, set it to 1234 because that is the default password and it's hopefully not production
if "PGPASSWORD" not in os.environ:
    os.environ["PGPASSWORD"] = "1234"
# If no database is set, set it to tsdb because that is the default database and it's hopefully not production
if "PGDATABASE" not in os.environ:
    os.environ["PGDATABASE"] = "ecallisto_tsdb"

##
CONNECTION = f' dbname={os.environ["PGDATABASE"]} user={os.environ["PGUSER"]} host={os.environ["PGHOST"]} password={os.environ["PGPASSWORD"]}'
# Map between seconds of timebucket and view name
CONT_AGG_VALUES = [60]
CONT_AGG_VALUES_VIEW_NAMES = ['1min']

def fill_missing_timesteps_with_nan(df):
    """
    Fill missing timesteps in a pandas DataFrame with NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to fill missing timesteps in.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with missing timesteps filled with NaN values.

    Notes
    -----
    This function is useful when working with time-series data that has missing timesteps.
    By filling the missing timesteps with NaN values, the DataFrame can be easily visualized
    or analyzed without introducing errors due to missing data.

    The function calculates the median time delta of the input DataFrame, and then creates
    a new index with evenly spaced values based on that delta. It then uses the pandas
    `reindex` function to fill in missing timesteps with NaN values.

    Examples
    --------
    >>> dates = pd.date_range('2023-02-19 01:00', '2023-02-19 05:00', freq='2H')
    >>> freqs = ['10M', '20M', '30M']
    >>> data = np.random.randn(len(dates), len(freqs))
    >>> df = pd.DataFrame(data, index=dates, columns=freqs)
    >>> df = fill_missing_timesteps_with_nan(df)
    >>> print(df)

                            10M       20M       30M
    2023-02-19 01:00:00 -0.349636  0.004947  0.546848
    2023-02-19 03:00:00       NaN       NaN       NaN
    2023-02-19 05:00:00 -0.576182  1.222293 -0.416526
    """
    # Change index to datetime, if it's not already
    df.index = pd.to_datetime(df.index)
    time_delta = np.median(np.diff(df.index.values))
    time_delta = pd.Timedelta(time_delta)
    new_index = pd.date_range(df.index[0], df.index[-1], freq=time_delta)
    df = df.reindex(new_index)
    return df

def get_column_names_clean(
    column_names, columns_to_drop=["burst_type"], columns_to_add=[]
):
    """Get the column names of a table in the database.

    Args:
        column_names (list): List of column names to clean.
        columns_to_drop (list): List of column names to drop.
        columns_to_add (list): List of column names to add at the beginning.

    Returns:
        list: List of column names without "" around the frequencies and trailing zeros.
    """
    column_names = [name.replace('"', "") for name in column_names]
    # Remove trailing zeros
    column_names = [name.rstrip("0").rstrip(".") for name in column_names]
    column_names = [to_float_if_possible(name) for name in column_names]
    column_names = [name for name in column_names if name not in columns_to_drop]
    if len(columns_to_add) > 0:
        for column in columns_to_add:
            column_names.insert(0, column)
    return column_names
    

def sql_result_to_df(
    result, columns: list = None, meta_data: dict = None
):
    """
    Converts the given result from a SQL query to a pandas DataFrame.
    
    Parameters:
    - result: The result obtained from a SQL query. This could be a list of dictionaries, where each dictionary represents a row of data. 
              Alternatively, 'result' could be a DataFrame, in which case it will be processed directly. 
              Each key in the dictionary represents a column name, and the corresponding value represents the data in that column.
              
              
    - datetime_col (str, optional): Name of the column to be treated as the datetime. 
                                     If 'datetime', the function will convert the 'datetime' column to pandas datetime format.
                                     If 'time', the function will not do any conversion.
                                     Defaults to 'datetime'.


    - datetime_col (str, optional): Name of the column to be treated as the datetime. 
                                     If 'datetime', the function will convert the 'datetime' column to pandas datetime format.
                                     If 'time', the function will not do any conversion.
                                     Defaults to 'datetime'.

    - columns (list, optional): List of column names for the resulting DataFrame. 
                                If not specified, the function will attempt to infer the column names from the 'result' input.
                                If 'result' is a list of dictionaries, the function will use the keys of the dictionaries as column names.
                                If 'result' is something else (e.g., a list of lists), the function will generate default column names.

    - meta_data (dict, optional): Dictionary containing metadata for the DataFrame. 
                                  Each key-value pair in the dictionary will be stored in the DataFrame's 'attrs' attribute.
                                  This can be used to attach additional information to the DataFrame.
                                  For example, 'meta_data' might contain information about when and how the data was collected.

    Returns:
    - df (pandas.DataFrame): DataFrame containing the data from 'result'. The DataFrame's index will be set to the datetime column, 
                             and any columns that only contain NaN values will be dropped.
                             
    Raises:
    - ValueError: If 'datetime_col' is not 'datetime' or 'time'.
    """

    if not isinstance(result, pd.DataFrame):
        if isinstance(result[0], dict):
            columns = list(result[0].keys())
        else:
            columns = [f"column_{i}" for i in range(len(result[0]))]
    else:
        columns = result.columns
    df = pd.DataFrame(result)
    
    # Clean colmns 
    columns = get_column_names_clean(columns)
    df.columns = columns
    if 'datetime' in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif 'bucketed_time' in df.columns:
        df["datetime"] = pd.to_datetime(df["bucketed_time"])
        df = df.set_index("bucketed_time")
    else:
        raise ValueError("datetime_col must be either 'datetime' or 'time'")
    # make columns prettier if possible by removing trailing 0s.
    df.columns = [
        col if col != "0" else "0."
        for col in df.columns.astype(str).str.rstrip("0").str.rstrip(".")
    ]
    if meta_data:
        for key, value in meta_data.items():
            df.attrs[key] = value
    return df.dropna(how="all", axis=1)

def fetch_data_from_chunks_to_df(
        chunks,
):
    # Initialize an empty DataFrame
    final_df = pd.DataFrame()

    # Iterate over the chunks of results
    for chunk in chunks:
        # Do any necessary processing on the chunk...
        
        # Append the DataFrame to the final_df
        final_df = pd.concat([final_df, chunk])

    return final_df


def create_table_sql(table_name, columns):
    """
    Creates a table with the given name and columns
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""CREATE TABLE {table_name} (
                            id SERIAL PRIMARY KEY,
                            {columns}
                        );
                        """
        )
        conn.commit()
        cursor.close()


def table_to_hyper_table(instrument, datetime_column):
    """
    Creates a table in the hyper database with the same name and columns as the given table.
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT create_hypertable('{instrument}', '{datetime_column}'
                        );
                        """
        )


def create_table_datetime_primary_key_sql(table_name, columns, datetime_column):
    """
    Creates a table with the given name and columns
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""CREATE TABLE {table_name} (
                            {datetime_column} TIMESTAMP PRIMARY KEY,
                            {columns}
                        );
                        """
        )
        conn.commit()
        cursor.close()


def drop_table_sql(table_name):
    """
    Drops a table from the database if it exists
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""DROP TABLE IF EXISTS {table_name};
                        """
        )
        conn.commit()
        cursor.close()

def drop_materialized_view_sql(view_name):
    """
    Drops a materialized view from the database if it exists
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""DROP MATERIALIZED VIEW IF EXISTS {view_name};
                        """
        )
        conn.commit()
        cursor.close()

def get_table_names_sql():
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema='public'
                       AND table_type='BASE TABLE';
                       """
        )

        tuple_list = cursor.fetchall()
        return [tup[0] for tup in tuple_list]

def get_view_names_sql():
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema='public'
                       AND table_type='VIEW';
                       """
        )

        tuple_list = cursor.fetchall()
        return [tup[0] for tup in tuple_list]

def get_table_names_with_data_between_dates_sql(start_date, end_date):
    table_names = get_table_names_sql()
    tables_with_data = []

    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()

        for table_name in table_names:
            cursor.execute(
                f"""SELECT EXISTS (
                   SELECT 1
                   FROM {table_name}
                   WHERE datetime BETWEEN '{start_date}' AND '{end_date}'
                   LIMIT 1);
                """
            )
            has_data = cursor.fetchone()[0]
            if has_data:
                tables_with_data.append(table_name)
        
    return tables_with_data

def check_if_table_has_data_between_dates_sql(table_name, start_date, end_date, datetime_col = "datetime"):
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM {table_name}
                WHERE {datetime_col} BETWEEN '{start_date}' AND '{end_date}'
                LIMIT 1
            );
            """
        )
        has_data = cursor.fetchone()[0]
        return has_data
    
def add_new_column_default_value_sql(
    table_name, column_name, column_type, default_value
):
    """
    Adds a new column to the given table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""ALTER TABLE {table_name}
                        ADD COLUMN IF NOT EXISTS {column_name} {column_type} DEFAULT {default_value};
                        """
        )
        conn.commit()
        cursor.close()


def add_new_column_sql(table_name, column_name, column_type):
    """
    Adds a new column to the given table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""ALTER TABLE {table_name}
                        ADD COLUMN IF NOT EXISTS {column_name} {column_type};
                        """
        )
        conn.commit()
        cursor.close()

def drop_column_sql(table_name, column_name):
    """
    Drops a column from the given table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""ALTER TABLE {table_name}
                        DROP COLUMN IF EXISTS {column_name};
                        """
        )
        conn.commit()
        cursor.close()

def get_hypertable_sizes_sql():
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT hypertable_name, hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)::regclass)
                FROM timescaledb_information.hypertables;
                """
        )
        df = pd.DataFrame(
            cursor.fetchall(), columns=["hypertable_name", "hypertable_size (B)"]
        )
        df["hypertable_size (GB)"] = df["hypertable_size (B)"].apply(
            lambda x: x / 1024 / 1024 / 1024
        )
        df = df.sort_values(by="hypertable_size (GB)", ascending=False)

        return df


def truncate_table_sql(table_name):
    """
    Truncates a table from the database if it exists
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""TRUNCATE TABLE {table_name};
                        """
        )
        conn.commit()
        cursor.close()


def get_column_names_sql(table_name):
    query = f"""SELECT column_name
                       FROM information_schema.columns
                       WHERE table_name = '{table_name}'
                       """
    check_query_for_harmful_sql(query)
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(query)

        tuple_list = cursor.fetchall()
        tuple_list = [tup[0] for tup in tuple_list]
        tuple_list = sort_column_names(tuple_list)
        tuple_list = [
            f'"{tup}"' if tup not in ["datetime", "bucketed_time", "burst_type"] else tup
            for tup in tuple_list
        ]
        return tuple_list


def get_rolling_mean_sql(table, start_time, end_time, timebucket="1H"):
    """
    Returns the rolling mean between start and end time in the given table, timebucketed.
    """
    columns = get_column_names_sql(table)
    columns = [column for column in columns if column not in ["datetime", "burst_type"]]
    agg_function_sql = ",".join([f"avg({column}) AS {column}" for column in columns])
    query = f"SELECT time_bucket('{timebucket}', datetime) AS time, {agg_function_sql} FROM {table} WHERE datetime BETWEEN '{start_time}' AND '{end_time}' GROUP BY time ORDER BY time"

    with psycopg2.connect(CONNECTION) as conn:
        df = pd.read_sql(query, conn)
        df["time"] = df["time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        df = df.set_index("time")
        df = df.rolling("1H").mean()
        df = df.reset_index()
        df["time"] = df["time"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        )
        return df
    

def values_from_database_sql(
    table: str,
    start_datetime: str,
    end_datetime: str,
    columns: List[str] = None,
    columns_not_to_select: List[str] = ["datetime", "burst_type"],
    chunk_size = None,
    **kwargs,
):
    """
    Returns all values between start and end time in the given table, without any aggregation.
    """
    # Type checks
    if not isinstance(table, str):
        raise TypeError(f"'table' should be of str type, got {type(table).__name__}")
    
    if not table in get_table_names_sql():
        raise ValueError(f"Table {table} does not exist in the database")

    if columns is not None and not all(isinstance(column, str) for column in columns):
        raise TypeError("'columns' should be a list of str")

    # Check date
    try:
        dateutil_parse(start_datetime)
    except ValueError as e:
        raise ValueError(f"start_datetime error: {e}")
    
    try:
        dateutil_parse(end_datetime)
    except ValueError as e:
        raise ValueError(f"start_datetime error: {e}")

    if not isinstance(columns_not_to_select, list) or not all(
        isinstance(column, str) for column in columns_not_to_select
    ):
        raise TypeError("'columns_not_to_select' should be a list of str")

    if not columns:
        columns = get_column_names_sql(table)
        columns = [column for column in columns if column not in columns_not_to_select]

    # Query
    columns_sql = ",".join(columns)
    query = f"SELECT datetime, {columns_sql} FROM {table} WHERE datetime BETWEEN '{start_datetime}' AND '{end_datetime}'"

    return _execute_query_for_values(query, columns, chunk_size)

def timedelta_to_sql_timebucket_value(timedelta):
    # Convert to seconds
    seconds = timedelta.total_seconds()

    # Convert to SQL-compatible value
    if seconds >= 86400:  # More than 1 day
        days = seconds / 86400
        sql_value = f"{int(days)} d" if days.is_integer() else f"{days:.1f} d"
    elif seconds >= 3600:  # More than 1 hour
        hours = seconds / 3600
        sql_value = f"{int(hours)} h" if hours.is_integer() else f"{hours:.1f} h"
    elif seconds >= 60:  # More than 1 minute
        minutes = seconds / 60
        sql_value = (
            f"{int(minutes)} min" if minutes.is_integer() else f"{minutes:.1f} min"
        )
    else:
        sql_value = f"{int(seconds)} s"

    return sql_value

def create_view_name_aggregation(table_name, timebucket, agg_function, values_seconds: list[int] = CONT_AGG_VALUES):
    timebucket_seconds = timebucket_string_to_seconds(timebucket)
    # Check which view to use
    for i, value in enumerate(values_seconds):
        if timebucket_seconds >= value:
            timebucket = CONT_AGG_VALUES_VIEW_NAMES[i]
    view_name = f"{table_name}_{timebucket.replace(' ', '')}_{agg_function}"
    view_name = view_name.lower()
    return view_name

def timebucket_string_to_seconds(timebucket: str) -> int:
    """Convert a timebucket string to seconds."""
    # Remove all spaces in the string
    if timebucket is None: # Standard setting is 250ms
        return 0.25
    else:
        timebucket = timebucket.replace(" ", "").lower()
        if 'ms' in timebucket:
            return int(timebucket.replace("ms", "")) / 1000

        return pytimeparse_parse(timebucket)
    
def round_timebucket_to_closest_seconds(timebucket: str, values_seconds: list[int] = CONT_AGG_VALUES) -> str:
    seconds = timebucket_string_to_seconds(timebucket)
    closest_value = min(values_seconds, key=lambda x:abs(x-seconds))
    return f"{closest_value} s"

def create_continuous_aggregate_sql(
    table: str,
    view_name: Optional[str] = None,
    timebucket: str = "1 minute",
    agg_function: str = "MAX",
    exclude_columns: List[str] = ["datetime", "burst_type"],
    **kwargs,
):
    """
    Create a continuous aggregate materialized view for the specified table.
    """
    
    # Check if table exists
    if table not in get_table_names_sql():
        raise ValueError(f"Table {table} does not exist in the database")

    if agg_function not in ["MAX", "MIN"]:
        raise ValueError(f"Invalid aggregation function: {agg_function}")
    
    if view_name is None:
        view_name = create_view_name_aggregation(table, timebucket, agg_function)

    if view_name in get_view_names_sql():
        print(f"View {view_name} already exists")
        return None

    pattern = r"^\d+(\.\d+)?\D+$"
    if not re.match(pattern, timebucket):
        raise TypeError(
            f"'timebucket' should be in the form <value><unit>, got {timebucket}"
        )

    columns = get_column_names_sql(table)
    columns = [column for column in columns if column not in exclude_columns]

    if len(columns) == 0:
        print(f"Table {table} has no columns to aggregate")
        return None

    agg_columns_sql = ",".join(
        [f"{agg_function}({column}) AS {column}" for column in columns]
    )

    # Create continuous aggregate query
    query = f"""
    CREATE MATERIALIZED VIEW {view_name} WITH (timescaledb.continuous) AS
    SELECT time_bucket(INTERVAL '{timebucket}', datetime) AS bucketed_time,
           {agg_columns_sql}
    FROM {table}
    GROUP BY bucketed_time
    WITH NO DATA;
    """
    _execute_query(query, check_query=False)
    return view_name

def remove_policy_sql(view_name):
    remove_query = f"""
    SELECT remove_continuous_aggregate_policy('{view_name}');
    """
    _execute_query(remove_query, check_query=False)

def add_continuous_aggregate_policy_sql(
    view_name: str,
    schedule_interval: str = "15 minutes",
    start_offset: str = None,
    end_offset: str = None,
    **kwargs,
):
    # Convert None values to SQL NULL without quotes
    start_offset = 'NULL' if start_offset is None else f"'{start_offset}'"
    end_offset = 'NULL' if end_offset is None else f"'{end_offset}'"

    policy_query = f"""
        SELECT add_continuous_aggregate_policy('{view_name}',
                                                start_offset => {start_offset},
                                                end_offset => {end_offset},
                                                schedule_interval => INTERVAL '{schedule_interval}');
        """
    _execute_query(policy_query, check_query=False)


def refresh_continuous_aggregate(table_name):
    query = f"CALL refresh_continuous_aggregate('{table_name}', NULL, NULL);"
    _execute_query(query, check_query=False)

def _execute_query(query: str, check_query: bool = True):
    """
    Executes the given query.
    """
    # Check the query for harmful SQL
    if check_query:
        check_query_for_harmful_sql(query)

    with psycopg2.connect(CONNECTION) as conn:
        with conn.cursor() as cur:
            cur.execute(query)

def timebucket_values_from_database_sql(
    table: str,
    start_datetime: str,
    end_datetime: str,
    columns: List[str] = None,
    timebucket: str = "1H",
    agg_function: str = "avg",
    quantile_value: float = None,
    columns_not_to_select: List[str] = ["datetime", "bucketed_time", "burst_type"],
    datetime_col = "datetime",
    preaggregated: bool = True,
    chunk_size = None,
    **kwargs,
):
    """
    Returns all values between start and end time in the given table, timebucketed and aggregated.
    """
    # Type checks
    if not isinstance(table, str):
        raise TypeError(f"'table' should be of str type, got {type(table).__name__}")
    
    if not table in get_table_names_sql():
        raise ValueError(f"Table {table} does not exist in the database")

    if columns is not None and not all(isinstance(column, str) for column in columns):
        raise TypeError("'columns' should be a list of str")

    # Check date
    try:
        dateutil_parse(start_datetime)
    except ValueError as e:
        raise ValueError(f"start_datetime error: {e}")
    
    try:
        dateutil_parse(end_datetime)
    except ValueError as e:
        raise ValueError(f"start_datetime error: {e}")

    pattern = r"^\d+(\.\d+)?\D+$"
    if not re.match(pattern, timebucket):
        raise TypeError(
            f"'timebucket' should be in the form <value><unit>, got {timebucket}"
        )

    if agg_function not in {"MIN", "MAX"}:
        raise ValueError(
            f"'agg_function' should be one of 'MIN', 'MAX'. Got {agg_function}"
        )

    if not isinstance(columns_not_to_select, list) or not all(
        isinstance(column, str) for column in columns_not_to_select
    ):
        raise TypeError("'columns_not_to_select' should be a list of str")

    if quantile_value is not None and not isinstance(quantile_value, float):
        raise TypeError(
            f"'quantile_value' should be of float type, got {type(quantile_value).__name__}"
        )
    if preaggregated:
        # If preaggrated yes, check if the table exists
        timebucket = round_timebucket_to_closest_seconds(timebucket)
        # Logic to select the correct view
        if timebucket is not None and timebucket != "0.25 s":
            view_name = create_view_name_aggregation(table, timebucket, agg_function)
        else:
            view_name = table
        # Check that the table actually exists
        if not (view_name in get_view_names_sql() or view_name in get_table_names_sql()):
            LOGGER.info(f"View {view_name} does not exist in the database")
        # And that is has the data
        elif not check_if_table_has_data_between_dates_sql(view_name, start_datetime, end_datetime, datetime_col='bucketed_time'):
            LOGGER.info(f"View {view_name} has no data between {start_datetime} and {end_datetime}")
        else:
            table = view_name
            datetime_col = "bucketed_time"
            LOGGER.info(f"Using view {view_name} with timebucket {timebucket}")

    LOGGER.info(f"Using table {table}")
            
    if not columns:
        columns = get_column_names_sql(table)
        columns = [column for column in columns if column not in columns_not_to_select]

    if agg_function == "quantile" and quantile_value is None:
        LOGGER.error(
            "quantile_value must be specified when using agg_function 'quantile'"
        )

    if agg_function == "quantile":
        agg_function_sql = ",".join(
            [
                f"percentile_disc({quantile_value}) WITHIN GROUP (ORDER BY {column}) AS {column}"
                for column in columns
            ]
        )
    else:
        agg_function_sql = ",".join(
            [f"{agg_function}({column}) AS {column}" for column in columns]
        )

    # Query
    query = f"SELECT time_bucket('{timebucket}', {datetime_col}) AS time, {agg_function_sql} FROM {table} WHERE {datetime_col} BETWEEN '{start_datetime}' AND '{end_datetime}' GROUP BY time ORDER BY time"
    
    return _execute_query_for_values(query, columns, chunk_size=chunk_size)
   
def _execute_query_for_values(query, columns, chunk_size=None):
    """
    Executes the given query and returns the results as a list of dictionaries.
    """
    # Check the query for harmful SQL
    check_query_for_harmful_sql(query)

    with psycopg2.connect(CONNECTION) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [
                dict(zip(["datetime"] + columns, row)) for row in cur.fetchall()
            ]  # return list of dict
    

def get_min_max_datetime_from_table_sql(table_name) -> tuple:
    """
    Returns the minimum and maximum datetime from the given table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT MIN(datetime), MAX(datetime)
                       FROM {table_name};
                       """
        )

        return cursor.fetchone()


def get_distinct_dates_from_table_sql(table_name) -> list:
    """
    Returns a list of distinct dates (in 'YYYY-MM-DD' format) from the given table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT DISTINCT DATE_TRUNC('day', datetime) AS date
                       FROM {table_name};
                       """
        )

        return [row[0].strftime("%Y-%m-%d") for row in cursor.fetchall()]


def drop_values_between_two_dates_sql(table_name, start_time, end_time):
    """
    Drops all values between the given start and end time in the given table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""DELETE FROM {table_name}
                       WHERE datetime BETWEEN %s AND %s;
                       """,
            (start_time, end_time),
        )
        conn.commit()
        cursor.close()


def insert_values_sql(table_name, columns, values):
    """
    Inserts values into the given table. If they already exist, the value is skipped.
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""INSERT INTO {table_name} ({columns})
                        VALUES {values}
                        ON CONFLICT DO NOTHING;
                        """
        )
        conn.commit()
        cursor.close()



def drop_database_sql(database_name):
    """
    Drops a database from the database if it exists
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT *
                FROM pg_stat_activity
                WHERE datname = {database_name};
                SELECT	pg_terminate_backend (pid)
                FROM	pg_stat_activity
                WHERE	pg_stat_activity.datname = {database_name};
                DROP DATABASE IF EXISTS {database_name};
                        """
        )
        conn.commit()
        cursor.close()


def get_size_of_table(table_name):
    """
    Returns the size of the given table in MB
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT pg_size_pretty(pg_total_relation_size('{table_name}'));
                        """
        )
        size = cursor.fetchone()[0]
        return size


def vacuum_full_database():
    """
    VACUUMs the full database
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute("ROLLBACK;")  # Roll back any open transactions
        cursor.execute("VACUUM FULL;")
        conn.commit()
        cursor.close()


def get_size_of_database_sql():
    """
    Returns the size of the database in MB
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT pg_size_pretty(pg_database_size('tsdb'));
                        """
        )
        size = cursor.fetchone()[0]
        return size

def check_query_for_harmful_sql(query: str):
    harmful_patterns = [
        "DROP TABLE",
        "DELETE FROM",
        "UPDATE",
        "TRUNCATE TABLE",
        "ALTER TABLE",
        "CREATE TABLE",
        "DROP DATABASE",
        "EXEC",  # or "EXECUTE"
        "EXECUTE",
        "UNION SELECT",
        ";",  # semicolon
    ]
    if any(pattern in query.upper() for pattern in harmful_patterns):
        raise ValueError("Detected potentially harmful SQL pattern in the query.")
    return query


def sort_column_names(list):
    return sorted(list, key=lambda x: to_float_if_possible_else_number(x, -1000))


def is_float(element) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def to_float_if_possible(element):
    if is_float(element):
        return float(element)
    else:
        return element


def to_float_if_possible_else_number(element, number):
    if is_float(element):
        return float(element)
    else:
        return number
