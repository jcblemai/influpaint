cd datasets/Flusight-forecast-data
echo " ⚠️ ⚠️ ⚠️  Make sure fork jcblemai/Flusight-forecast-data is synced cdcepi/Flusight-forecast-data !!!"
git pull
cd ../..

cd datasets/delphi-epidata
git pull
cd ../..
cp datasets/delphi-epidata/src/client/delphi_epidata.py helpers/

cd datasets/synthetic/flu-scenario-modeling-hub/
git pull
cd ../../..

cd "West-Nile Virus/cdc-forecasting-challenge"
git pull
cd ../..

cd "West-Nile Virus/WNV-forecast-project-2023"
