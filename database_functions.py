import os
import re
from datetime import datetime
from typing import List

import pandas as pd
import psycopg2

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
def create_continuous_aggregate_view_no_refresh(table_name):
    view_name = f"{table_name}_daily_row_count"
    with psycopg2.connect(CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            CREATE MATERIALIZED VIEW {view_name}
            WITH (timescaledb.continuous) AS
            SELECT time_bucket('1 day', datetime) AS day,
                   COUNT(*) AS row_count
            FROM {table_name}
            GROUP BY day
            WITH NO DATA;
        """)
        conn.commit()

def get_daily_rows_for_table_sql(table_name):
    view_name = f"{table_name}_daily_row_count"
    with psycopg2.connect(CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {view_name};")
        results = cur.fetchall()
    return results
         
def remove_continuous_aggregate_policy(table_name):
    view_name = f"{table_name}_daily_row_count"

    with psycopg2.connect(CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT remove_continuous_aggregate_policy('{view_name}');
        """)
        conn.commit()

def recompute_continuous_aggregate_view(table_name):
    view_name = f"{table_name}_daily_row_count"
    with psycopg2.connect(CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT refresh_continuous_aggregate('{view_name}');
        """)
        conn.commit()
        
def set_daily_refresh_policy(table_name):
    view_name = f"{table_name}_daily_row_count"
    with psycopg2.connect(CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT add_continuous_aggregate_policy('{view_name}',
                start_offset => INTERVAL '20 year',
                end_offset => INTERVAL '0 min',
                schedule_interval => INTERVAL '1 day');
        """)
        conn.commit()

def drop_daily_rows_for_table_sql(table_name):
    view_name = f"{table_name}_daily_row_count"
    with psycopg2.connect(CONNECTION) as conn:
        cur = conn.cursor()
        cur.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name};")
        conn.commit()

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


def count_rows_of_data_per_day_per_table_sql(table_name):
    """
    Returns the number of rows per day per table
    """
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""SELECT date_trunc('day', datetime) as day, COUNT(*)
                        FROM {table_name}
                        GROUP BY day
                        ORDER BY day;
                        """
        )
        tuple_list = cursor.fetchall()
        return [tup for tup in tuple_list]
     
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


def insert_is_burst_status_between_dates_sql(tablename, start_date, end_date):
    """Insert is_burst status between two dates.

    Parameters
    ----------
    tablename : str
        The table name to insert the is_burst status for.
    start_date : `~datetime.datetime`
        The start date to insert the is_burst status for.
    end_date : `~datetime.datetime`
        The end date to insert the is_burst status for.


    Returns
    -------
    None

    Notes
    -----
    The function first finds the unique index data and the indices of the non-unique index data.
    It then combines the non-unique index data using the method specified by the `method` parameter.
    """
    start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
        UPDATE {tablename}
        SET is_burst = TRUE
        WHERE datetime BETWEEN '{start_date}' AND '{end_date}'
        """
        )
        conn.commit()
        cursor.close()


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
            f'"{tup}"' if tup not in ["datetime", "burst_type"] else tup
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
    **kwargs,
):
    """
    Returns all values between start and end time in the given table, without any aggregation.
    """
    # Type checks
    if not isinstance(table, str):
        raise TypeError(f"'table' should be of str type, got {type(table).__name__}")

    if columns is not None and not all(isinstance(column, str) for column in columns):
        raise TypeError("'columns' should be a list of str")
    
    # Check date
    datetime_format = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(start_datetime, datetime_format)
    except ValueError:
        raise ValueError("start_datetime should be a string in the format 'YYYY-MM-DD HH:MM:SS'")

    try:
        datetime.strptime(end_datetime, datetime_format)
    except ValueError:
        raise ValueError("end_datetime should be a string in the format 'YYYY-MM-DD HH:MM:SS'")

    if not isinstance(columns_not_to_select, list) or not all(isinstance(column, str) for column in columns_not_to_select):
        raise TypeError("'columns_not_to_select' should be a list of str")
    
    if not columns:
        columns = get_column_names_sql(table)
        columns = [column for column in columns if column not in columns_not_to_select]

    # Query
    columns_sql = ",".join(columns)
    query = f"SELECT datetime, {columns_sql} FROM {table} WHERE datetime BETWEEN '{start_datetime}' AND '{end_datetime}'"
    
    # Check the query for harmful SQL
    check_query_for_harmful_sql(query)

    with psycopg2.connect(CONNECTION) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [dict(zip(['datetime'] + columns, row)) for row in cur.fetchall()]  # return list of dict


def timebucket_values_from_database_sql(
    table: str,
    start_datetime: str,
    end_datetime: str,
    columns: List[str] = None,
    timebucket: str = "1H",
    agg_function: str = "avg",
    quantile_value: float = None,
    columns_not_to_select: List[str] = ["datetime", "burst_type"],
    **kwargs,
):
    """
    Returns all values between start and end time in the given table, timebucketed and aggregated.
    """
    # Type checks
    if not isinstance(table, str):
        raise TypeError(f"'table' should be of str type, got {type(table).__name__}")

    if columns is not None and not all(isinstance(column, str) for column in columns):
        raise TypeError("'columns' should be a list of str")
    
    # Check date
    datetime_format = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(start_datetime, datetime_format)
    except ValueError:
        raise ValueError("start_datetime should be a string in the format 'YYYY-MM-DD HH:MM:SS'")

    try:
        datetime.strptime(end_datetime, datetime_format)
    except ValueError:
        raise ValueError("end_datetime should be a string in the format 'YYYY-MM-DD HH:MM:SS'")

    pattern = r"^\d+(\.\d+)?\D+$"
    if not re.match(pattern, timebucket):
        raise TypeError(f"'timebucket' should be in the form <value><unit>, got {timebucket}")

    if agg_function not in {"MIN", "MAX", "AVG", "MEDIAN"}:
        raise ValueError(f"'agg_function' should be one of 'MIN', 'MAX', 'AVG', 'MEDIAN'. Got {agg_function}")

    if not isinstance(columns_not_to_select, list) or not all(isinstance(column, str) for column in columns_not_to_select):
        raise TypeError("'columns_not_to_select' should be a list of str")
    
    if quantile_value is not None and not isinstance(quantile_value, float):
        raise TypeError(f"'quantile_value' should be of float type, got {type(quantile_value).__name__}")
    
    if not columns:
        columns = get_column_names_sql(table)
        columns = [column for column in columns if column not in columns_not_to_select]

    if agg_function == "quantile" and quantile_value is None:
        raise ValueError(
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
    query = f"SELECT time_bucket('{timebucket}', datetime) AS time, {agg_function_sql} FROM {table} WHERE datetime BETWEEN '{start_datetime}' AND '{end_datetime}' GROUP BY time ORDER BY time"
    
    # Check the query for harmful SQL
    check_query_for_harmful_sql(query)

    with psycopg2.connect(CONNECTION) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [dict(zip(['datetime'] + columns, row)) for row in cur.fetchall()]  # return list of dict


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
            (start_time, end_time)
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


def get_values_from_database_sql(table, start_time, end_time, columns=None):
    """
    Returns the values from the given table between the given start and end time
    """
    if columns is None:
        columns = "*"

    with psycopg2.connect(CONNECTION) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {columns} FROM {table} WHERE datetime BETWEEN '{start_time}' AND '{end_time}'"
            )
            return cur.fetchall()

def sql_background_image_to_df(result, columns=None, meta_data: dict = None):
    """
    Converts the given result from a sql query to a pandas dataframe
    """
    df = pd.DataFrame(result, columns=columns)
    df = df.set_index("time")
    for column in df.columns:
        try:
            # Check if int is possible
            if all(df[column].astype(int) == df[column]):
                df[column] = df[column].astype(int)
            else:
                df[column] = df[column].astype(float)
        except:
            pass
    if meta_data:
        for key, value in meta_data.items():
            df.attrs[key] = value
    return df


from typing import List

import psycopg2


def get_spectogram_background_image_sql(
    table: str,
    end_time: str,
    length: str = "1w",
    columns: List[str] = None,
    timebucket: str = "hour",
    agg_function: str = "avg",
    quantile_value: float = None,
    columns_not_to_select: List[str] = ["datetime", "burst_type"],
):
    """
    Get a background image for the spectrogram plot
    :param table: table name
    :param end_time: end time
    :param timebucket: timebucket. e.g. minute, hour. This is then average over the timebucket between start and end time. E.g. if you select 1h, it takes the
    <agg_function> of all daily-hours between start and end time, thus returning a row per hour.
    :param agg_function: aggregation function. e.g. MAX, MIN, AVG, SUM, QUANTILE
    :param quantile_value: quantile value. Only used if agg_function is quantile
    :param columns_not_to_select: columns not to select
    :return: background image
    """
    if not columns:
        columns = get_column_names_sql(table)
        columns = [column for column in columns if column not in columns_not_to_select]

    if agg_function == "quantile" and quantile_value is None:
        raise ValueError(
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

    # Get data between start and end time, grouped by daily hour and aggregated by the agg_function with the help of DATE_TRUNC
    query = f"""
    SELECT
        DATE_PART('{timebucket}', datetime) as time, {agg_function_sql}
    FROM
        {table}
    WHERE 
        datetime BETWEEN '{end_time}'::timestamp - '{length}'::interval AND '{end_time}'::timestamp
    GROUP BY
        time
    """
    with psycopg2.connect(CONNECTION) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

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
