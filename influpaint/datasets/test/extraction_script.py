# How to recreate test dataset from main dataset - MINIMAL for fast testing
import pandas as pd

# Load full dataset
all_datasets_df = pd.read_parquet('Flusight/flu-datasets/all_datasets.parquet')

print("Extracting minimal test dataset for frame 880 duplicate bug...")

# 1. PROBLEMATIC FRAME: The exact frame that caused duplicates (frame 880)
problematic_frame = all_datasets_df[
    (all_datasets_df['datasetH2'] == 'round5_SigSci-SWIFT_A-2024-08-01') & 
    (all_datasets_df['sample'] == '1.0_NA_101') &
    (all_datasets_df['fluseason'] == 2024) &
    # Exclude location '11' to force intelligent filling
    (all_datasets_df['location_code'] != '11')
]

# 2. FILL SOURCE DATA: SMH data with location '11' for lookup (multiple sources)
smh_data = all_datasets_df[all_datasets_df['datasetH1'] == 'SMH_R4-R5']

# Include multiple H2 datasets for randomization testing
fill_sources = []
for h2_pattern in ['GLEAM', 'flepiMoP', 'USC-SIkJalpha']:
    matching_h2 = [h2 for h2 in smh_data['datasetH2'].unique() if h2_pattern in h2][:2]  # Max 2 per pattern
    for h2 in matching_h2:
        h2_data = smh_data[
            (smh_data['datasetH2'] == h2) & 
            (smh_data['location_code'] == '11') & 
            (smh_data['fluseason'] == 2023)  # Different year for intelligent fill
        ].head(53)  # One full season per H2
        if not h2_data.empty:
            fill_sources.append(h2_data)

# 3. FLEPI DATA: Small subset for 100M config
flepi_data = all_datasets_df[
    (all_datasets_df['datasetH1'] == 'flepiR1') & 
    (all_datasets_df['fluseason'] == 2022)
].head(200)  # Just enough for one frame

# 4. Original fluview data for testing (both 2010 and 2021)
fluview_2010 = all_datasets_df[
    (all_datasets_df['datasetH1'] == 'fluview') & 
    (all_datasets_df['fluseason'] == 2010)
].head(500)  # 2010 data with some missing locations for testing

fluview_2021 = all_datasets_df[
    (all_datasets_df['datasetH1'] == 'fluview') & 
    (all_datasets_df['fluseason'] == 2021)
]  # 2021 data - get complete dataset for fill source and skip test

# 5. Add flusurv data for multiplier tests
flusurv_data = all_datasets_df[
    (all_datasets_df['datasetH1'] == 'flusurv') & 
    (all_datasets_df['fluseason'] == 2021)
].head(50)

# Combine all minimal data
test_dataset = pd.concat([
    problematic_frame,
    *fill_sources,
    flepi_data,
    fluview_2010,
    fluview_2021,
    flusurv_data
], ignore_index=True)

test_dataset.to_parquet('influpaint/datasets/test/test_dataset.parquet', index=False)

print(f"✅ Created MINIMAL test dataset with {len(test_dataset)} rows")
print(f"   - Problematic frame: {len(problematic_frame)} rows (missing location 11)")
print(f"   - Fill sources: {sum(len(fs) for fs in fill_sources)} rows (location 11 available)")
print(f"   - FlepiR1 data: {len(flepi_data)} rows")
print(f"   - Fluview 2010: {len(fluview_2010)} rows (missing locations 12, 22)")
print(f"   - Fluview 2021: {len(fluview_2021)} rows (complete)")
print(f"   - Flusurv data: {len(flusurv_data)} rows")
print(f"   - Available H2 datasets with location 11:")
for fs in fill_sources:
    if not fs.empty:
        print(f"     * {fs['datasetH2'].iloc[0]}: {len(fs)} rows")
print("🎯 Dataset optimized for FAST frame 880 duplicate bug testing")
