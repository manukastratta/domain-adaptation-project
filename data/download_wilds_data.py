# from wilds import get_dataset

# # Load the full dataset, and download it if necessary
# dataset = get_dataset(dataset="fmow", download=True)

from pathlib import Path
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import tarfile
import datetime
import pytz
from PIL import Image
from tqdm import tqdm
#from wilds.common.utils import subsample_idxs
#from wilds.common.metrics.all_metrics import Accuracy
#from wilds.common.grouper import CombinatorialGrouper
#from wilds.datasets.wilds_dataset import WILDSDataset

Image.MAX_IMAGE_PIXELS = 10000000000


categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]


class FMoWDataset():

    def __init__(self, version=None, root_dir='data/', download=False, split_scheme='official', seed=111, use_ood_val=True):
        self._version = version
        #self._data_dir = self.initialize_data_dir(root_dir, download)
        self._data_dir = root_dir

        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}
        self._source_domain_splits = [0, 1, 2]

        self.oracle_training_set = False
        if split_scheme == 'official':
            split_scheme = 'time_after_2016'
        elif split_scheme == 'mixed-to-test':
            split_scheme = 'time_after_2016'
            self.oracle_training_set = True
        self._split_scheme = split_scheme

        self.root = Path(self._data_dir)
        self.seed = int(seed)
        self._original_resolution = (224, 224)

        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}

        self.metadata = pd.read_csv(self.root / 'rgb_metadata.csv')
        country_codes_df = pd.read_csv(self.root / 'country_code_mapping.csv')
        countrycode_to_region = {k: v for k, v in zip(country_codes_df['alpha-3'], country_codes_df['region'])}
        regions = [countrycode_to_region.get(code, 'Other') for code in self.metadata['country_code'].to_list()]
        self.metadata['region'] = regions
        all_countries = self.metadata['country_code']

        self.num_chunks = 101
        self.chunk_size = len(self.metadata) // (self.num_chunks - 1)

        if self._split_scheme.startswith('time_after'):
            year = int(self._split_scheme.split('_')[2])
            year_dt = datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)
            self.test_ood_mask = np.asarray(pd.to_datetime(self.metadata['timestamp'], format='mixed') >= year_dt)
            print("HEY: self.test_ood_mask: ", self.test_ood_mask)
            # use 3 years of the training set as validation
            year_minus_3_dt = datetime.datetime(year-3, 1, 1, tzinfo=pytz.UTC)
            self.val_ood_mask = np.asarray(pd.to_datetime(self.metadata['timestamp'], format='mixed') >= year_minus_3_dt) & ~self.test_ood_mask
            self.ood_mask = self.test_ood_mask | self.val_ood_mask

            print(self.val_ood_mask.shape)
        else:
            raise ValueError(f"Not supported: self._split_scheme = {self._split_scheme}")

        self._split_array = -1 * np.ones(len(self.metadata))
        for split in self._split_dict.keys():
            idxs = np.arange(len(self.metadata))
            if split == 'test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                idxs = idxs[self.test_ood_mask & test_mask]
            elif split == 'val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                idxs = idxs[self.val_ood_mask & val_mask]
            elif split == 'id_test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                idxs = idxs[~self.ood_mask & test_mask]
            elif split == 'id_val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                idxs = idxs[~self.ood_mask & val_mask]
            else:
                split_mask = np.asarray(self.metadata['split'] == split)
                idxs = idxs[~self.ood_mask & split_mask]

            if self.oracle_training_set and split == 'train':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                unused_ood_idxs = np.arange(len(self.metadata))[self.ood_mask & ~test_mask]
                subsample_unused_ood_idxs = subsample_idxs(unused_ood_idxs, num=len(idxs)//2, seed=self.seed+2)
                subsample_train_idxs = subsample_idxs(idxs.copy(), num=len(idxs) // 2, seed=self.seed+3)
                idxs = np.concatenate([subsample_unused_ood_idxs, subsample_train_idxs])
            self._split_array[idxs] = self._split_dict[split]
            print("HEY self._split_dict[split]: ", self._split_dict[split])

        if not use_ood_val:
            self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
            self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

        # filter out sequestered images from full dataset
        seq_mask = np.asarray(self.metadata['split'] == 'seq')
        # take out the sequestered images
        self._split_array = self._split_array[~seq_mask] # self._split_array:  [-1. -1. -1. ... -1.  2.  2.]. of len: 470086
        print("len self._split_array: ", len(self._split_array))
        self.full_idxs = np.arange(len(self.metadata))[~seq_mask] # len: 470086. [     0      1      2 ... 470083 470084 470085]

        self._y_array = np.asarray([self.category_to_idx[y] for y in list(self.metadata['category'])])
        self.metadata['y'] = self._y_array
        self._y_array = torch.from_numpy(self._y_array).long()[~seq_mask]
        self._y_size = 1
        self._n_classes = 62

        # convert region to idxs
        all_regions = list(self.metadata['region'].unique())
        region_to_region_idx = {region: i for i, region in enumerate(all_regions)}
        self._metadata_map = {'region': all_regions}
        print("self._metadata_map: ", self._metadata_map)
        region_idxs = [region_to_region_idx[region] for region in self.metadata['region'].tolist()]
        self.metadata['region'] = region_idxs

        # make a year column in metadata
        year_array = -1 * np.ones(len(self.metadata))
        ts = pd.to_datetime(self.metadata['timestamp'], format='mixed')
        for year in range(2002, 2018):
            year_mask = np.asarray(ts >= datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)) \
                        & np.asarray(ts < datetime.datetime(year+1, 1, 1, tzinfo=pytz.UTC))
            year_array[year_mask] = year - 2002
        self.metadata['year'] = year_array
        self._metadata_map['year'] = list(range(2002, 2018))
        print("self._metadata_map['year']: ", self._metadata_map['year'])

        self._metadata_fields = ['region', 'year', 'y']
        self._metadata_array = torch.from_numpy(self.metadata[self._metadata_fields].astype(int).to_numpy()).long()[~seq_mask]

        # Create dataframes for train, validation, and test splits
        # trim the seq from metadata. should be of len 470086
        old_metadata = self.metadata.copy()
        self.metadata = self.metadata.iloc[0: len(self._split_array)]
        def add_filename(row):
            name = row.name
            return f"rgb_img_{name}.png"
        # TRAIN
        train_mask = np.asarray(self._split_array == self._split_dict["train"])
        train_df = self.metadata[train_mask]
        train_df["new_filename"] = train_df.apply(add_filename, axis=1)
        # VAL OOD
        val_ood_mask = np.asarray(self._split_array == self._split_dict["val"])
        val_ood_df = self.metadata[val_ood_mask]
        val_ood_df["new_filename"] = val_ood_df.apply(add_filename, axis=1)
        # TEST OOD
        test_ood_mask = np.asarray(self._split_array == self._split_dict["test"])
        test_ood_df = self.metadata[test_ood_mask]
        test_ood_df["new_filename"] = test_ood_df.apply(add_filename, axis=1)
        # VAL ID
        val_id_mask = np.asarray(self._split_array == self._split_dict["id_val"])
        val_id_df = self.metadata[val_id_mask]
        val_id_df["new_filename"] = val_id_df.apply(add_filename, axis=1)
        # TEST ID
        test_id_mask = np.asarray(self._split_array == self._split_dict["id_test"])
        test_id_df = self.metadata[test_id_mask]
        test_id_df["new_filename"] = test_id_df.apply(add_filename, axis=1)
        
        
        train_df.to_csv(self.root / "train.csv")
        val_ood_df.to_csv(self.root / "val.csv")    
        test_ood_df.to_csv(self.root / "test.csv")
        val_id_df.to_csv(self.root / "id_val.csv")
        test_id_df.to_csv(self.root / "id_test.csv")

        selected_rows = len(train_df) + len(val_ood_df) + len(test_ood_df) + len(val_id_df) + len(test_id_df)  # 141696
        subtract = len(old_metadata) - seq_mask.sum()   # 470086
        # assert selected_rows == subtract
        
        # Sanity check: no overlapping images between train_df, val_ood_df, and test_ood_df
        train_s = set(list(train_df["new_filename"]))
        assert len(train_s) == len(train_df)
        val_ood_df_s = set(list(val_ood_df["new_filename"]))
        assert len(val_ood_df_s) == len(val_ood_df)
        test_ood_df_s = set(list(test_ood_df["new_filename"]))
        assert len(test_ood_df_s) == len(test_ood_df)

        s = set((train_s.union(val_ood_df_s)).union(test_ood_df_s)) # removing all possible duplicates
        assert len(s) == (len(train_df) + len(val_ood_df) + len(test_ood_df))

        return
        

        self._eval_groupers = {
            'year': CombinatorialGrouper(dataset=self, groupby_fields=['year']),
            'region': CombinatorialGrouper(dataset=self, groupby_fields=['region']),
        }



        #super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        idx = self.full_idxs[idx]
        #img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        img = None
        return img



dataset = FMoWDataset(root_dir="fmow_v1.1")