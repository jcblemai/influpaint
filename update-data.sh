cd Flusight/Flusight-forecast-data
#echo " ⚠️ ⚠️ ⚠️  Make sure fork jcblemai/Flusight-forecast-data is synced cdcepi/Flusight-forecast-data !!!"
git pull
cd ../..

cd Flusight/flu-datasets/delphi-epidata
git pull
cd ../../..
cp Flusight/flu-datasets/delphi-epidata/src/client/delphi_epidata.py helpers/

cd Flusight/flu-datasets/synthetic/flu-scenario-modeling-hub/
git pull
cd ../../../..

cd Flusight/FluSight-forecast-hub
git pull
cd ../..
