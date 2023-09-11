from database_functions import create_continuous_aggregate_sql, get_table_names_sql, drop_materialized_view_sql
from tqdm import tqdm


drop_materialized_view_sql('alaska_anchorage_01_1minute_max')


for table in tqdm(get_table_names_sql()):
    view_name = table + '_daily_row_count'
    assert '_daily_row_count' in view_name
    drop_materialized_view_sql(view_name)


for table in tqdm(get_table_names_sql()):
    create_continuous_aggregate_sql(table, timebucket='1minute', agg_function='MAX')