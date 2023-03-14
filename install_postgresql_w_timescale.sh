
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
# conda env config vars set PGPASSWORD=<> (maybe not secure, check.)
# change to this user with sudo -u ecallisto -s
# Clone the repo and cd into it