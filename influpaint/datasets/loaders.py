import pandas as pd
import numpy as np
import torch
import xarray as xr
from . import read_datasources


class FluDataset(torch.utils.data.Dataset):
    """
    transform_enrich are for enriching the dataset and are not inverted (thus must have a mean effect of zero).
    """
    def __init__(self, flu_dyn, transform=None, transform_enrich=None, transform_inv=None, channels=3):
        """
        Args:
            flu_dyn (np.array): flu dynamics, shape (n_samples, n_features, n_dates, n_places)
        """
        self.transform = transform
        self.transform_inv = transform_inv
        self.transform_enrich = transform_enrich

        self.flu_dyn = flu_dyn
        #  self.max_per_feature = self.flu_dyn.max(dim=["date", "place", "sample"])
        self.max_per_feature = np.max(
            self.flu_dyn, axis=(0, 2, 3)
        )  # TODO: Check with channels, also perhaps use keepdims=True for broadcasting

        print(
            f"created dataset with max {np.array(self.max_per_feature)}, full dataset has shape {self.flu_dyn.shape}"
        )


    @classmethod
    def from_SMHR1_fluview(
        cls, season_setup, download=False, transform=None, transform_inv=None, channels=3
    ):
        netcdf_file = (
            "Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc"
        )
        channels = 1
        flu_dyn = xr.open_dataarray(netcdf_file)
        if channels == 1:
            flu_dyn = flu_dyn.sel(feature="incidH_FluA") + flu_dyn.sel(
                feature="incidH_FluB"
            )
            flu_dyn = flu_dyn.expand_dims("feature", axis=1).assign_coords(
                feature=("feature", ["incidH"])
            )
        flu_dyn1 = flu_dyn.data

        fluview = read_datasources.get_from_epidata(
            dataset="fluview", season_setup=season_setup, download=download, write=False
        )
        df = fluview[fluview["location_code"].isin(season_setup.locations)]
        flu_dyn2 = np.array(read_datasources.dataframe_to_arraylist(df=df, season_setup=season_setup))

        flu_dyn2 = flu_dyn2.repeat(90, axis=0)
        print(
            f"After repeat, fluview data has shape {flu_dyn2.shape} vs {flu_dyn1.shape} from csp"
        )
        flu_dyn = np.concatenate((flu_dyn1, flu_dyn2), axis=0)

        rng = np.random.default_rng()

        rng.shuffle(flu_dyn, axis=0)  # this is inplace

        return cls(
            flu_dyn=flu_dyn,
            transform=transform,
            transform_inv=transform_inv,
            channels=channels,
        )

    @classmethod
    def from_csp_SMHR1(
        cls, netcdf_file, transform=None, transform_inv=None, channels=3
    ):
        flu_dyn = xr.open_dataarray(netcdf_file)
        if channels == 1:
            flu_dyn = flu_dyn.sel(feature="incidH_FluA") + flu_dyn.sel(
                feature="incidH_FluB"
            )
            flu_dyn = flu_dyn.expand_dims("feature", axis=1).assign_coords(
                feature=("feature", ["incidH"])
            )
        flu_dyn = flu_dyn.data
        return cls(
            flu_dyn=flu_dyn,
            transform=transform,
            transform_inv=transform_inv,
            channels=channels,
        )
    @classmethod
    def from_synthetic_dataset(
        cls, netcdf_file, transform=None, transform_inv=None, channels=3
    ):
        flu_dyn = xr.open_dataarray(netcdf_file)
        flu_dyn = flu_dyn.data[:, :channels, :, :]  # Select the right number of channels
        return cls(
            flu_dyn=flu_dyn,
            transform=transform,
            transform_inv=transform_inv,
            channels=channels,
        )

    @classmethod
    def from_xarray(
        cls, netcdf_file, transform=None, transform_inv=None, channels=1
    ):
        """
        Load dataset from xarray NetCDF file.
        
        Args:
            netcdf_file: Full path to NetCDF file
            transform: Transform function
            transform_inv: Inverse transform function  
            channels: Number of channels
            
        Returns:
            FluDataset instance
        """
        flu_dyn = xr.open_dataarray(netcdf_file)
        flu_dyn = flu_dyn.data[:, :channels, :, :]  # Select the right number of channels
        return cls(
            flu_dyn=flu_dyn,
            transform=transform,
            transform_inv=transform_inv,
            channels=channels,
        )

    @classmethod
    def from_flusurvCSP(cls, season_setup, transform=None, transform_inv=None, channels=3):
        csp_flusurv = pd.read_csv(
            "Flusight/flu-datasets/flu_surv_cspGT.csv", parse_dates=["date"]
        )
        df = pd.merge(
            csp_flusurv,
            season_setup.locations_df,
            left_on="FIPS",
            right_on="abbreviation",
            how="left",
        )
        df["fluseason"] = df["date"].apply(season_setup.get_fluseason_year)
        df["fluseason_fraction"] = df["date"].apply(season_setup.get_fluseason_fraction)
        flu_dyn = np.array(
            read_datasources.dataframe_to_arraylist(
                df, season_setup=season_setup, value_column="incidH"
            )
        )
        return cls(
            flu_dyn=flu_dyn,
            transform=transform,
            transform_inv=transform_inv,
            channels=channels,
        )

    @classmethod
    def from_fluview(
        cls, season_setup, download=False, transform=None, transform_inv=None, channels=3
    ):
        fluview = read_datasources.get_from_epidata(
            dataset="fluview", season_setup=season_setup, download=download, write=False
        )
        df = fluview[fluview["location_code"].isin(season_setup.locations)]
        flu_dyn = np.array(read_datasources.dataframe_to_arraylist(df=df, season_setup=season_setup))

        return cls(
            flu_dyn=flu_dyn,
            transform=transform,
            transform_inv=transform_inv,
            channels=channels,
        )

    def add_transform(self, transform, transform_inv, transform_enrich, bypass_test=False):
        self.transform = transform
        self.transform_inv = transform_inv
        self.transform_enrich = transform_enrich
        if not bypass_test:
            # test that the inverse transform really works
            self.test(0)

    def __len__(self):
        return self.flu_dyn.shape[0]

    def __getitem__(self, idx):
        frame = self.get_sample_transformed_enriched(idx=idx)
        return torch.from_numpy(frame).float()
    
    def get_sample_raw(self, idx):
        if torch.is_tensor(idx):
            idxl = idx.tolist()
        else:
            idxl = idx
        # should be squeezed when channel = 3 ?
        epi_frame = self.flu_dyn[idxl, :, :, :]
        return epi_frame
    
    def get_sample_transformed(self, idx):
        return self.apply_transform(self.get_sample_raw(idx))
    
    def get_sample_transformed_enriched(self, idx):
        return self.apply_transform(self.apply_enrich(self.get_sample_raw(idx)))
    
    def apply_transform(self, epi_frame):
        if self.transform:
            array = self.transform(epi_frame)
            return array
        else:
            return epi_frame
        
    def apply_enrich(self, epi_frame):
        if self.transform_enrich:
            array = self.transform_enrich(epi_frame)
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
        epi_frame_n = self.get_sample_transformed(idx)
        assert (
            np.abs(self.apply_transform_inv(epi_frame_n) - self.flu_dyn[idx]) < 1e-5
        ).all()
        print("test passed: back and forth transformation are ok ✅")
