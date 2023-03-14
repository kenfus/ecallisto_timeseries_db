# Timescale DB for e-callisto data
This Repository contains Python code provides a connection to a TimescaleDB, a time-series database built on top of PostgreSQL. It allows users to store and analyze time-series data in a scalable and efficient manner. The code utilizes the psycopg2 library to establish a connection between the Python script and the TimescaleDB database.


## Timescale DB
The database is a timeseries database, where each row is a timestamp and each column is a frequency. For each instrument, a table is created with the name of the instrument. How the data is inserted and how the table is created can be found in the file `database_functions.py`.

### Schema of the Database
The database contains a table for each instrument. The name of the table is the name of the instrument. The table contains a column for each frequency. The name of the column is the frequency. The table also contains a column for the timestamp. The timestamp is the index of the table. 

### Functionality
The Python code first creates variables that set the connection parameters for the database such as the host, username, and password. If a password has not been set, the script sets the password to the default password of 1234. These variables are then used to construct the CONNECTION string, which is passed to psycopg2 to establish a connection to the TimescaleDB database.

Once the connection is established, users can use pandas, a popular data manipulation library, to interact with the data in the TimescaleDB database. By utilizing TimescaleDB, users can take advantage of its built-in time-series optimizations and benefit from faster data retrieval and analysis.

### Installation
A TimescaleDB database can be installed on a Linux machine using the following commands and instructions.
```bash	
## Timescale db
sudo apt install gnupg postgresql-common apt-transport-https lsb-release wget
./usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update
sudo apt install timescaledb-2-postgresql-14

sudo apt-get update
sudo apt-get install postgresql-client
sudo systemctl restart postgresql

# continue with this:
# https://docs.timescale.com/install/latest/self-hosted/installation-linux/#setting-up-the-timescaledb-extension-on-debian-based-systems


## Start Database
# Create user ecallisto and switch to it with "useradd -m ecallisto"
# conda env config vars set PGUSER=ecallisto
# conda env config vars set PGPASSWORD=<> (maybe not secure, check.)
# change to this user with sudo -u ecallisto -s
# Clone the repo and cd into it

```
## Instrument Data Addition to Timescale DB
This function adds instrument data to a Timescale database with the index being a timestamp and columns being the frequencies. The database contains data for instruments that contain the specified substring in their name, starting from the given start_date. The data is added iteratively with a chunk size of chunk_size and utilizing cpu_count CPU cores. The main code to add data to the database is in the file `continiously_add_data_to_database.py`. The main function of it is called `main` and is described below.

### Parameters
- start_date : datetime.date - The starting date for adding instrument data to the database.
- instrument_substring : str - A substring to match instrument names with.
- chunk_size : int - The number of instrument data files to add to the database at once.
- cpu_count : int - The number of CPU cores to use when adding instrument data to the database.

### Notes
- This function iteratively adds instrument data to a database.
- First, it creates a database using the data for the current day and the previous day.
- Then, it adds the data to the database.
- Next, it iterates over all tables in the database, and for each table, it adds data for the previous days_chunk_size days.
- The function stops when no new data is added to the database.

### Examples
To add instrument data to a database for instruments containing the substring 'ALASKA-COHOE' starting from January 1st, 2023, with a days chunk size of 30, a chunk size of 100, and 8 CPU cores, you could run:

```python
start_date = datetime.date(2023, 1, 1)
instrument_substring = 'ALASKA-COHOE'
days_chunk_size = 30
chunk_size = 100
cpu_count = 8
main(start_date, instrument_substring, days_chunk_size, chunk_size, cpu_count)
```

### Continously Add Data to Database
This function is also found in the file `continiously_add_data_to_database.py`. It is called `continiously_add_data_to_database` and is described below.