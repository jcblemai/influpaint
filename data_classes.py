import pandas as pd
import numpy as np
import torch 
import xarray as xr


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