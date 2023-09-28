from tqdm import tqdm

from database_functions import (
    add_continuous_aggregate_policy_sql,
    create_continuous_aggregate_sql,
    drop_materialized_view_sql,
    get_table_names_sql,
    get_view_names_sql,
    remove_policy_sql,
)

for view in tqdm(get_view_names_sql()):
    remove_policy_sql(view)
    add_continuous_aggregate_policy_sql(
        view, schedule_interval="15 minutes", start_offset="8 weeks"
    )
