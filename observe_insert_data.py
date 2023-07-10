import os
import time
from datetime import datetime, timedelta
import logging_utils 
from bulk_load_to_database_between_dates import add_specs_from_paths_to_database

LOGGER = logging_utils.setup_custom_logger("observe_insert_data_tsdb")

def get_files_with_timestamps(path):
    try:
        files_with_timestamps = {}
        if os.path.exists(path):  # Check if the path exists
            for dirpath, _, filenames in os.walk(path):
                for file in filenames:
                    full_path = os.path.join(dirpath, file)
                    files_with_timestamps[full_path] = os.path.getmtime(full_path)
        else:
            LOGGER.error(f"Path {path} does not exist (yet?).")
        return files_with_timestamps
    except Exception as e:
        LOGGER.error(f"Error during fetching timestamps: {str(e)}")
        return {}

def get_dirs_to_monitor(base_path, days_to_check):
    dirs_to_monitor = []
    try:
        for i in range(days_to_check):
            day_ago = datetime.now() - timedelta(days=i)
            date_path = day_ago.strftime('%Y/%m/%d')
            # Append to the beginning of list
            dirs_to_monitor.insert(0, f"{base_path}{date_path}/")

        LOGGER.info(f"Monitoring {len(dirs_to_monitor)} directories.")
        LOGGER.info(f"First and last directory to monitor: {dirs_to_monitor[0]} and {dirs_to_monitor[-1]}.")
        return dirs_to_monitor
    except Exception as e:
        LOGGER.error(f"Error during fetching directories to monitor: {str(e)}")
        return []

def monitor_directories(base_path, days_to_check):
    dirs_to_monitor = get_dirs_to_monitor(base_path, days_to_check)
    prev_state = {dir: get_files_with_timestamps(dir) for dir in dirs_to_monitor}
    current_day = datetime.now().date()

    try:
        while True:
            new_day = datetime.now().date()
            if new_day != current_day:
                LOGGER.info(f"New day! {new_day}")
                current_day = new_day
                
                oldest_day = (datetime.now() - timedelta(days=days_to_check)).strftime('%Y/%m/%d')
                oldest_dir = f"{base_path}{oldest_day}/"
                LOGGER.info(f"Removing {oldest_dir} from prev_state")

                new_day_dir = f"{base_path}{new_day.strftime('%Y/%m/%d')}/"
                LOGGER.info(f"Adding {new_day_dir} to prev_state")

                del prev_state[oldest_dir]
                dirs_to_monitor.remove(oldest_dir)
                dirs_to_monitor.append(new_day_dir)
                
                prev_state[new_day_dir] = get_files_with_timestamps(new_day_dir)

            LOGGER.info(f"Example previous: {list(prev_state.keys())[-1]} has {len(list(prev_state.values())[0])} files.")

            curr_state = {dir: get_files_with_timestamps(dir) for dir in dirs_to_monitor if dir in prev_state}
            LOGGER.info(f"Example current: {list(curr_state.keys())[-1]} has {len(list(curr_state.values())[0])} files.")

            added = []
            removed = []
            modified = []

            for dir in dirs_to_monitor:
                added += [f for f in curr_state[dir] if f not in prev_state[dir]]
                removed += [f for f in prev_state[dir] if f not in curr_state[dir]]
                modified += [f for f in curr_state[dir] if f in prev_state[dir] and curr_state[dir][f] != prev_state[dir][f]]

            if added:
                added_examples = ', '.join(added[:3]) + ('...' if len(added) > 3 else '')
                LOGGER.info(f"In {dir} - Added ({len(added)}): {added_examples}")
            if removed:
                removed_examples = ', '.join(removed[:3]) + ('...' if len(removed) > 3 else '')
                LOGGER.info(f"In {dir} - Removed ({len(removed)}): {removed_examples}")
            if modified:
                modified_examples = ', '.join(modified[:3]) + ('...' if len(modified) > 3 else '')
                LOGGER.info(f"In {dir} - Modified ({len(modified)}): {modified_examples}")

            to_add = added + modified
            if to_add:
                LOGGER.info(f"Adding {len(to_add)} files to the database...")
                add_specs_from_paths_to_database(to_add)
                LOGGER.info(f"Done adding {len(to_add)} files to the database.")

            prev_state = curr_state
            time.sleep(3*60)

    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt received. Exiting...")
        return

if __name__ == "__main__":
    base_path = "/mnt/nas05/data01/radio/2002-20yy_Callisto/"
    days_to_check = 30
    try:
        monitor_directories(base_path, days_to_check)
    except Exception as e:
        LOGGER.error(f"Fatal error during directory monitoring: {str(e)}")