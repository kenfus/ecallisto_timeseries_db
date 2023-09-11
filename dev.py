from database_functions import create_continuous_aggregate_sql, get_table_names_sql, drop_materialized_view_sql, get_view_names_sql, add_continuous_aggregate_policy_sql, remove_policy_sql
from tqdm import tqdm


for view in tqdm(get_view_names_sql()):
    remove_policy_sql(view)
    add_continuous_aggregate_policy_sql(view, schedule_interval='15 minutes', start_offset='8 weeks')