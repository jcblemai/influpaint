set -o xtrace

# 1. Update official flusight repositories
cd Flusight/2022-2023/FluSight-forecast-hub-official
git pull
cd ../../..

cd Flusight/2023-2024/FluSight-forecast-hub-official
git pull
cd ../../..


cd Flusight/2024-2025/FluSight-forecast-hub-official
git pull
cd ../../..


cd Flusight/flu-datasets/delphi-epidata
git pull
cd ../../..
cp Flusight/flu-datasets/delphi-epidata/src/client/delphi_epidata.py helpers/

#cd Flusight/flu-datasets/synthetic/flu-scenario-modeling-hub/
#git pull
#cd ../../../..
