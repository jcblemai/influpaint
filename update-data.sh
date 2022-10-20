cd datasets/Flusight-forecast-data
echo " !!!! Make sure fork jcblemai/Flusight-forecast-data is synced cdcepi/Flusight-forecast-data !!!"
git pull
cd ../..
cd datasets/delphi-epidata
git pull
cd ../..
cp datasets/delphi-epidata/src/client/delphi_epidata.py helpers/
