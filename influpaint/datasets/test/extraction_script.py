# How to recreate test dataset from main dataset
import pandas as pd

# Load full dataset
all_datasets_df = pd.read_parquet('Flusight/flu-datasets/all_datasets.parquet')

# Extract test subset
fluview_data = all_datasets_df[all_datasets_df['datasetH1'] == 'fluview']
fluview_2010 = fluview_data[
    (fluview_data['fluseason'] == 2010) & 
    (~fluview_data['location_code'].isin(['12', '22']))
]
fluview_2021 = fluview_data[fluview_data['fluseason'] == 2021]

flusurv_data = all_datasets_df[all_datasets_df['datasetH1'] == 'flusurv']
flusurv_subset = flusurv_data[flusurv_data['fluseason'] == 2015]

smh_data = all_datasets_df[all_datasets_df['datasetH1'] == 'SMH_R4-R5']
smh_subset = smh_data[
    (smh_data['datasetH2'] == 'round4_USC-SIkJalpha_A-2023-08-14') & 
    (smh_data['fluseason'] == 2023) &
    (smh_data['sample'].isin(['1', '2']))
]

test_dataset = pd.concat([fluview_2010, fluview_2021, flusurv_subset, smh_subset], ignore_index=True)
test_dataset.to_parquet('test/test_dataset.parquet', index=False)
