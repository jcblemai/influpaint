import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
import epiweeks
import torch # todo: separate dataset classes from data utils
import xarray as xr

def padto64x64(x):
    return np.pad(x, ((0, 64-x.shape[0]), (0, 64-x.shape[1])), mode='constant', constant_values=0)

flu_season_start_date = pd.to_datetime("2020-12-15")

# locations, in the right order
flusight_locations = pd.read_csv("datasets/Flusight-forecast-data/data-locations/locations.csv")
flusight_locations['geoid'] = flusight_locations['location']+'000'
flusight_locations = flusight_locations.iloc[1:,:].reset_index(drop=True)  # skip first row, which is the US full
flusight_locations['location_code'] = flusight_locations['location']       # "location" collides with datasets column name
flusight_locations.drop(columns=['location'], inplace=True)




def get_location_name(location_code):
    return flusight_locations[flusight_locations['location_code']==location_code]['location_name'].values[0]

def get_fluseason_year(ts):
    if ts.dayofyear >= flu_season_start_date.dayofyear:
        return ts.year 
    else:
        return ts.year - 1

def get_fluseason_fraction(ts):
    if ts.dayofyear >= flu_season_start_date.dayofyear:
        return (ts.dayofyear - flu_season_start_date.dayofyear) / 365
    else:
        return ((ts.dayofyear + 365) - flu_season_start_date.dayofyear)  / 365

def get_all_locations(dataset):
    if dataset == "flusurv":
        locations_fn = "datasets/delphi-epidata/labels/flusurv_locations.txt"
        locations = pd.read_csv(locations_fn, sep='\t', header=None, names=['location'])["location"].to_list()
    elif dataset=="fluview":
        import importlib  
        fluview_locations_m = importlib.import_module("datasets.delphi-epidata.src.acquisition.fluview.fluview_locations")
        fll_dict = fluview_locations_m.cdc_to_delphi
        locations = []
        for region_type in fll_dict.keys():
            for region_name, flloc in fll_dict[region_type].items():
                locations.append(flloc)
    return locations


def get_from_epidata(dataset, locations="all", write=True, download=True):
    """ 
    Read a dataset from epidata. Each dataset is a dataframe with columns:
    - 'week_enddate' (datetime)
    - 'location' (str) original location name
    - 'location_code' (str) location name in the format used by the flusight data
    - 'value' (float) the value of interest
    - 'fluseason' (int) the flu season (e.g. 2019)
    - 'fluseason_fraction' (float) the fraction of the flu season (e.g. 0.5 for the middle of the season)
    """

    if dataset == "flusurv" or dataset == "fluview":
        if download:
            # by location otherwise queries is too big
            df_list = []
            if locations == "all":
                locations = get_all_locations(dataset=dataset)

            for location in locations:
                if dataset == 'flusurv':
                    res = Epidata.flusurv(location, [Epidata.range(190001, 202251)])  # large range to get all data
                elif dataset == "fluview":
                    res = Epidata.fluview(location, [Epidata.range(190001, 202251)])  # large range to get all data
                if res['result'] == 1:
                    flu_data_loc = pd.json_normalize(res['epidata'])
                    print(f">> {location: <12} {res['result']}, {res['message']}, with {len(res['epidata']):4} data points from {flu_data_loc.epiweek.min()} to {flu_data_loc.epiweek.max()}")
                    df_list.append(flu_data_loc)
                else:
                    print(f"EE {location: <12} {res['result']}, {res['message']} !")
            
            df = pd.concat(df_list)
            df['week_enddate'] = df['epiweek'].astype(str).apply(lambda x: epiweeks.Week.fromstring(week_string=x, system="cdc").enddate())

        else:
            df = pd.read_csv(f"datasets/{dataset}.csv")
    elif dataset == "flusight":
        df = pd.read_csv("datasets/Flusight-forecast-data/data-truth/truth-Incident Hospitalizations.csv", parse_dates=True, index_col='date')
        df['week_enddate'] = df.index
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    df['week_enddate'] = pd.to_datetime(df['week_enddate'])

    if write:  # write before merge
        df.to_csv(f"datasets/{dataset}.csv", index=False)
    
    # merge with locations, taking care of new york
    if dataset == "flusurv": 
        df["location_tomerge"] = df["location"]
        df["location_tomerge"] = df["location_tomerge"].str.replace("NY_albany", "NY")
        df["location_tomerge"] = df["location_tomerge"].str.replace("NY_rochester", "NY")
        right_on = "abbreviation"
        value_col = 'rate_overall'
    elif dataset == "fluview": 
        df["location_tomerge"] = df['region'].str.upper()
        df["location_tomerge"] = df["location_tomerge"].str.replace("jfk".upper(), "NY")
        df["location_tomerge"] = df["location_tomerge"].str.replace("ny_minus_jfk".upper(), "NY")
        right_on = "abbreviation"
        value_col = "ili"
    elif dataset == "flusight":
        print("/!\ Make sure ./update_data.sh is ran AND that the fork is updated")
        df["location_tomerge"] = df["location"]
        right_on = "location_code"
        value_col = "value"
    df = pd.merge(df, flusight_locations, left_on="location_tomerge", right_on=right_on, how='left')
    df.drop(columns=['location_tomerge'], inplace=True)

    # get the flu season year and it's fraction elapsed
    df["fluseason"]= df["week_enddate"].apply(get_fluseason_year)
    df["fluseason_fraction"]= df["week_enddate"].apply(get_fluseason_fraction)

    return df

class FluDynamicsDataset1D(torch.utils.data.Dataset):
    def __init__(self, dataset_type, download=False, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.dataset_type = dataset_type


        self.transform = transform

        if dataset_type == "onlyfluview":
            self.samples = []
            fluview = get_from_epidata(dataset="fluview", download=download, write=False)
            df = fluview[fluview['location_code'].isin(flusight_locations.location_code)]
            df_piv = df.pivot(columns='location_code', values='ili', index=["fluseason", "fluseason_fraction"])
            for season in df_piv.index.unique(level='fluseason'):
                array = df_piv.loc[season][flusight_locations.location_code].to_numpy()
                array[np.isnan(array)] = 0
                self.samples.append(np.array([padto64x64(array)])) # need one dimension for feature/channel
                
        elif dataset_type == "all":
            self.flu_dyn = {
                "flusurv": get_from_epidata(dataset="flusurv", download=download, write=False),
                "fluview": get_from_epidata(dataset="fluview", download=download, write=False),
                "flusight": get_from_epidata(dataset="flusight", download=download, write=False),
            }
        
        #if channels = 1:
        #    self.flu_dyn=self.flu_dyn.isel(feature=0)
        
        # let's store the min and max
        #self.max_per_feature = self.flu_dyn.max(dim=["date", "place", "sample"])
        #print(f"created dataset with scale {np.array(self.max_per_feature)}")
        #self.flu_dyn_norm = (np.sqrt(self.flu_dyn)/np.sqrt(self.max_per_feature))*2#-1
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame, idx = self.getitem_nocast(idx)
        return torch.from_numpy(frame).float(), idx
    
    def getitem_nocast(self, idx):
        if torch.is_tensor(idx):
            idxl = idx.tolist()
        else:
            idxl=idx
            
        epi_frame = self.samples[idxl]#.astype(np.float32)

        #shoudle be a dict ?
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            epi_frame = self.transform(epi_frame)

        return epi_frame, idx
    
    def unnormalized(self, array):
        return array
    
    def test(self, idx):
        """
        test that we can transform and go back & get the same thing
        """
        epi_frame_n, idx_n = self.getitem_nocast(idx)
        assert idx_n == idx
        assert (np.abs(self.unnormalized(epi_frame_n) - self.flu_dyn.sel(sample=idx)) < 1e-3).all()


class FluDynamicsSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, netcdf_file, transform=None):
        """
        Args:
            netcdf_file (string): Path to the netcdf file with the xarray data
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.flu_dyn = xr.open_dataarray(netcdf_file)
        self.transform = transform
        
        #if channels = 1:
        #    self.flu_dyn=self.flu_dyn.isel(feature=0)
        
        # let's store the min and max
        self.max_per_feature = self.flu_dyn.max(dim=["date", "place", "sample"])
        print(f"created dataset with scale {np.array(self.max_per_feature)}")
        self.flu_dyn_norm = (np.sqrt(self.flu_dyn)/np.sqrt(self.max_per_feature))*2#-1
        
        
    def __len__(self):
        return len(self.flu_dyn.sample)
    
    def __getitem__(self, idx):
        frame, idx = self.getitem_nocast(idx)
        return torch.from_numpy(frame).float(), idx
    
    def getitem_nocast(self, idx):
        if torch.is_tensor(idx):
            idxl = idx.tolist()
        else:
            idxl=idx
            
        epi_frame = self.flu_dyn_norm.sel(sample=idxl).squeeze().to_numpy()#.astype(np.float32)

        #shoudle be a dict ?
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            epi_frame = self.transform(epi_frame)

        return epi_frame, idx
    
    def unnormalized(self, array):
        #return (((array)/3)*np.sqrt((self.max_per_feature.to_numpy())[:,np.newaxis, np.newaxis]))**2
        return (((array)/2)*np.sqrt((self.max_per_feature.to_numpy())[:,np.newaxis, np.newaxis]))**2
        #return ((array+.75)*np.sqrt((self.max_per_feature.to_numpy())[:,np.newaxis, np.newaxis]))**2
    
    def test(self, idx):
        """
        test that we can transform and go back & get the same thing
        """
        epi_frame_n, idx_n = self.getitem_nocast(idx)
        assert idx_n == idx
        #print(self.unnormalized(epi_frame_n).shape, self.unnormalized(epi_frame_n)[0,0,0])
        #print(self.flu_dyn.sel(sample=idx).shape, self.flu_dyn.sel(sample=idx)[0,0,0])
        assert (np.abs(self.unnormalized(epi_frame_n) - self.flu_dyn.sel(sample=idx)) < 1e-3).all()
    

def randomscale_transforms(image, max, min):
    import random
    scale = random.uniform(min, max)
    return image*scale