import pandas as pd
import numpy as np
import torch 
import xarray as xr
import data_utils

class FluDataset(torch.utils.data.Dataset):
    def __init__(self, flu_dyn, transform=None, transform_inv=None, channels=3):
        """
        Args:
            flu_dyn (np.array): flu dynamics, shape (n_samples, n_features, n_dates, n_places)
        """
        self.transform = transform
        self.transform_inv = transform_inv

        self.flu_dyn = flu_dyn
        #  self.max_per_feature = self.flu_dyn.max(dim=["date", "place", "sample"])
        self.max_per_feature = np.max(self.flu_dyn, axis=(0,2,3)) # TODO: Check with channels, also perhaps use keepdims=True for broadcasting

        print(f"created dataset with max {np.array(self.max_per_feature)}, full dataset has shape {self.flu_dyn.shape}")

    @classmethod
    def from_SMHR1_fluview(cls,
                    flusetup,
                    download=False,
                    transform=None, 
                    transform_inv=None, 
                    channels=3):
        netcdf_file = 'Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc'
        channels = 1
        flu_dyn = xr.open_dataarray(netcdf_file)
        if channels == 1:
            flu_dyn = flu_dyn.sel(feature="incidH_FluA")+ flu_dyn.sel(feature="incidH_FluB")
            flu_dyn = flu_dyn.expand_dims('feature', axis=1).assign_coords(feature=('feature', ['incidH']))
        flu_dyn1 = flu_dyn.data

        fluview = data_utils.get_from_epidata(dataset="fluview", flusetup=flusetup, download=download, write=False)
        df = fluview[fluview['location_code'].isin(flusetup.locations)]
        flu_dyn2 = np.array(data_utils.dataframe_to_arraylist(df=df, flusetup=flusetup))

        flu_dyn2 = flu_dyn2.repeat(90, axis=0)
        print(f"After repeat, fluview data has shape {flu_dyn2.shape} vs {flu_dyn1.shape} from csp")
        flu_dyn = np.concatenate((flu_dyn1, flu_dyn2), axis=0)

        rng = np.random.default_rng()

        rng.shuffle(flu_dyn, axis=0) # this is inplace

        return cls(flu_dyn=flu_dyn, transform=transform, transform_inv=transform_inv, channels=channels)

    @classmethod
    def from_csp_SMHR1(cls, 
                    netcdf_file, 
                    transform=None, 
                    transform_inv=None,
                    channels=3):
        flu_dyn = xr.open_dataarray(netcdf_file)
        if channels == 1:
            flu_dyn = flu_dyn.sel(feature="incidH_FluA")+ flu_dyn.sel(feature="incidH_FluB")
            flu_dyn = flu_dyn.expand_dims('feature', axis=1).assign_coords(feature=('feature', ['incidH']))
        flu_dyn = flu_dyn.data
        return cls(flu_dyn=flu_dyn, transform=transform, transform_inv=transform_inv, channels=channels)

    @classmethod
    def from_flusurvCSP(cls, 
                    flusetup,
                    transform=None, 
                    transform_inv=None, 
                    channels=3):
        csp_flusurv = pd.read_csv("datasets/flu_surv_cspGT.csv", parse_dates=["date"])
        df = pd.merge(csp_flusurv, flusetup.locations_df, left_on="FIPS", right_on="abbreviation", how='left')
        df["fluseason"]= df["date"].apply(flusetup.get_fluseason_year)
        df["fluseason_fraction"]= df["date"].apply(flusetup.get_fluseason_fraction)
        flu_dyn = np.array(data_utils.dataframe_to_arraylist(df, flusetup = flusetup, value_column='incidH'))
        return cls(flu_dyn=flu_dyn, transform=transform, transform_inv=transform_inv, channels=channels)

    @classmethod
    def from_fluview(cls, 
                    flusetup,
                    download=False,
                    transform=None, 
                    transform_inv=None, 
                    channels=3):
        fluview = data_utils.get_from_epidata(dataset="fluview", flusetup=flusetup, download=download, write=False)
        df = fluview[fluview['location_code'].isin(flusetup.locations)]
        flu_dyn = np.array(data_utils.dataframe_to_arraylist(df=df, flusetup=flusetup))

        return cls(flu_dyn=flu_dyn, transform=transform, transform_inv=transform_inv, channels=channels)


    def add_transform(self, transform, transform_inv, bypass_test=False):
        self.transform = transform
        self.transform_inv = transform_inv
        if not bypass_test:
            # test that the inverse transform really works
            self.test(0)
        
    def __len__(self):
        return self.flu_dyn.shape[0]
    
    def __getitem__(self, idx):
        frame = self.getitem_nocast(idx)
        return torch.from_numpy(frame).float()
    
    def getitem_nocast(self, idx):
        if torch.is_tensor(idx):
            idxl = idx.tolist()
        else:
            idxl=idx
        # should be squeezed when channel = 3 ?
        epi_frame = self.flu_dyn[idxl,:,:,:]
        epi_frame = self.apply_transform(epi_frame)
        return epi_frame

    def apply_transform(self, epi_frame):
        if self.transform:
            array = self.transform(epi_frame)
            return array
        else:
            return epi_frame
    
    def apply_transform_inv(self, epi_frame):
        if self.transform_inv:
            array = self.transform_inv(epi_frame)
            return array
        else:
            return epi_frame
    
    def test(self, idx):
        """
        test that we can transform and go back & get the same thing
        """
        epi_frame_n = self.getitem_nocast(idx)
        assert (np.abs(self.apply_transform_inv(epi_frame_n) - self.flu_dyn[idx]) < 1e-5).all()
        print("test passed: back and forth transformation are ok ✅")

# These transform applies to a numpy object with dimensions
#  (feature, date, place)

def transform_randomscale(image, max, min):
    import random
    scale = random.uniform(min, max) # TODO should not be uniform !!!
    return image*scale

def transform_channelwisescale(image, scale): # TODO write for three channel like it was above
    return image*scale

def transform_channelwisescale_inv(image, scale):
    return image/scale

def transform_sqrt(image):
    return np.sqrt(image)

def transform_sqrt_inv(image):
    return image**2

def transform_shift(image, shift=-1):
    return image+shift

def transform_shift_inv(image, shift=-1):
    return image-shift

def transform_rollintime(image, shift):
    r_val = np.roll(image, shift=shift, axis=1)
    return r_val

def transform_random_rollintime(image, min_shift, max_shift):
    import random
    shift = random.randint(min_shift, max_shift)
    return transform_rollintime(image, shift)

def transform_random_padintime(image, min_shift, max_shift, neutral_value=0):
    import random
    shift = random.randint(min_shift, max_shift)
    r_val = transform_rollintime(image, shift)
    if shift >= 0:
        r_val[:,:shift] = neutral_value
    else:
        r_val[:,shift:] = neutral_value
    
    return r_val

def transform_randomnoise(image, sigma=0.2):
    mu = 1
    return image * np.random.normal(mu, sigma, image.shape)

from scipy.stats import skewnorm

def transform_skewednoise(image, scale=.4, a=-1):
    r = skewnorm.rvs(loc=1, scale=scale, a = a, size=image.shape)
    r[r<0] = 0
    return image * r
