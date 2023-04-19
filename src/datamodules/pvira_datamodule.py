from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import numpy as np
import json
from src.datamodules.components.datautils import cvt_sin_cos_to_phase, cvt_phase_to_sin_cos
from matplotlib import pyplot as plt
from einops import rearrange

class PviraResultDatasetTest(Dataset):
    def __init__(self, test_json, sincos_form=False):
        self.test_json = test_json
        self.sincos_form = sincos_form
        with open(test_json) as f:
            self.test_data = json.load(f)
        print(f"Loaded {len(self.test_data)} test samples")

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        img_flow = np.load(self.test_data[idx]['npy'])
        # unpack
        if self.sincos_form:
            img1_stack = img_flow[0:3]
            img1_mag = img_flow[3]
            img2_stack = img_flow[4:7]
            img2_mag = img_flow[7]
            flow = img_flow[8:]

            img1_sc = np.stack(cvt_phase_to_sin_cos(img1_stack), axis=1)
            img2_sc = np.stack(cvt_phase_to_sin_cos(img2_stack), axis=1)
            img1_sc = rearrange(img1_sc, 'c a h w d -> (c a) h w d')  # c=3 three phase, a=2 sin and cos
            img2_sc = rearrange(img2_sc, 'c a h w d -> (c a) h w d')
            img1 = np.concatenate((img1_sc, img_flow[3:4, ...]), axis=0)  # concatenate mag. C=2+2+2+1
            img2 = np.concatenate((img2_sc, img_flow[7:8, ...]), axis=0)

        img1, img2 = torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

        return img1, img2

class PviraResultDataset(Dataset):
    def __init__(self, data_dir, Nframes = 2, transform=None, max_gap=8, portion=1.0, sincos_form=False,
                 train_json=None, isCache=False):
        self.data_dir = data_dir
        self.transform = transform
        self.max_gap = max_gap
        self.portion = portion
        self.sincos_form = sincos_form
        self.train_json = train_json
        self.Nframes = Nframes
        self.isCache = isCache

        self.jdicts = self.load_json()
        self.data_pair=[]
        self.data_pair_idx = []
        self.data_cache = []

        if self.Nframes == 2:
            self.get_data_pair() # populate self.data_pair
        else:
            self.get_data_pair_multiframes()

        if self.isCache:
            self.cache_data()
        print("Data loaded, total: ", len(self.data_pair))

    def cache_data(self):
        """cache data and do preprocessing"""
        print("Caching data...")
        for i, jdict in enumerate(self.jdicts):
            img = np.load(jdict['npy'])
            img_sin, img_cos = cvt_phase_to_sin_cos(img[:3, ...])
            img_sc = np.stack((img_sin, img_cos), axis=1)
            img_sc = rearrange(img_sc, 'c a h w d -> (c a) h w d')
            img_sc_m = np.concatenate((img_sc, img[3:, ...]), axis=0)  # concatenate mag. C=2+2+2+1
            img_sc_m = np.ascontiguousarray(img_sc_m)
            img_sc_m = torch.from_numpy(img_sc_m).float()
            self.data_cache.append(img_sc_m)

    def load_json(self):
        with open(self.data_dir +'/'+ self.train_json) as f:
            jdicts = json.load(f)
        return jdicts

    def get_data_pair(self):
        """populate self.data_pair"""
        for i, jdict1 in enumerate(self.jdicts):
            for j, jdict2 in enumerate(self.jdicts):
                if (jdict1['subj_id'] == jdict2['subj_id']) \
                        and (jdict1['utterance'] == jdict2['utterance']) \
                        and (jdict1['frame'] != jdict2['frame']) \
                        and (abs(jdict1['frame'] - jdict2['frame']) <= self.max_gap):
                    self.data_pair.append((jdict1['npy'], jdict2['npy'])) # tuple of (fixed, moving)
                    self.data_pair_idx.append([i, j])
        self.data_pair = self.data_pair[:int(len(self.data_pair)*self.portion)]

    def get_data_pair_multiframes(self):
        """populate self.data_pair"""
        for i, jdict_st in enumerate(self.jdicts):
            for j, jdict_end in enumerate(self.jdicts):
                if (jdict_st['subj_id'] == jdict_end['subj_id']) \
                        and (jdict_st['utterance'] == jdict_end['utterance']) \
                        and (jdict_end['frame'] - jdict_st['frame'] >= 2) \
                        and (jdict_end['frame'] - jdict_st['frame'] <= self.max_gap):
                    # order matters 1-3-6, instead of 6-3-1
                    Nmframes = self.Nframes - 2
                    Nincre = abs(jdict_st['frame'] - jdict_end['frame']) // (self.Nframes - 1)
                    Mindices = [i + k * Nincre for k in range(1, Nmframes + 1)]
                    Mframes = [self.jdicts[m]['npy'] for m in Mindices]
                    self.data_pair.append([jdict_st['npy'],] + Mframes + [jdict_end['npy'],])
                    self.data_pair_idx.append([i,] + Mindices + [j,])
        # reverse the order of each seq to augment data
        self.data_pair = self.data_pair + [list(reversed(pair)) for pair in self.data_pair]
        self.data_pair_idx = self.data_pair_idx + [list(reversed(pair)) for pair in self.data_pair_idx]
        print(len(self.data_pair))

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        if self.Nframes == 2:
            img1p, img2p = self.data_pair[idx]
            img1, img2 = np.load(img1p), np.load(img2p) # shape: (64, 64, 64, 4) first three channel are phase images, last are mag
            # preprocessing # TODO: unify with multiframe processing
            img1_sin, img1_cos = cvt_phase_to_sin_cos(img1[:3, ...])
            img2_sin, img2_cos = cvt_phase_to_sin_cos(img2[:3, ...])
            img1_sc = np.stack((img1_sin, img1_cos), axis=1)
            img2_sc = np.stack((img2_sin, img2_cos), axis=1)
            img1_sc = rearrange(img1_sc, 'c a h w d -> (c a) h w d')  # c=3 three phase, a=2 sin and cos
            img2_sc = rearrange(img2_sc, 'c a h w d -> (c a) h w d')
            img1 = np.concatenate((img1_sc, img1[3:, ...]), axis=0)  # concatenate mag. C=2+2+2+1
            img2 = np.concatenate((img2_sc, img2[3:, ...]), axis=0)
            img1 = np.ascontiguousarray(img1)
            img2 = np.ascontiguousarray(img2)
            img1, img2 = torch.from_numpy(img1).float(), torch.from_numpy(img2).float()
            return img1, img2
        else: # multiframes
            if self.isCache:
                imgs = [self.data_cache[i] for i in self.data_pair_idx[idx]]
                return imgs
            else:
                imgs = [np.load(imgp) for imgp in self.data_pair[idx]]
                # preprocessing
                imgs_sincos = [cvt_phase_to_sin_cos(img[:3, ...]) for img in imgs]
                imgs_sc = [np.stack((img_sin, img_cos), axis=1) for img_sin, img_cos in imgs_sincos]
                imgs_sc = [rearrange(img_sc, 'c a h w d -> (c a) h w d') for img_sc in imgs_sc]
                imgs = [np.concatenate((img_sc, img[3:, ...]), axis=0) for img_sc, img in zip(imgs_sc, imgs)]
                imgs = [np.ascontiguousarray(img) for img in imgs]
                imgs = [torch.from_numpy(img).float() for img in imgs]
                return imgs



class PVIRADataModule(LightningDataModule):
    """A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
            """
    def __init__(
        self,
        data_dir: str = "data/",
        Nframes: int = 2,
        train_val_test_split: Tuple[float,float,float] = (0.4, 0.4, 0.2),
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        portion: float = 1.0,
        sincos_form: bool = False,
        max_gap: int = 8,
        train_json: str="/munge/pvira_results_info.json",
        test_json: Optional[str] = None,
            isCache: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations: None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = PviraResultDataset(
                self.hparams.data_dir,
                Nframes=self.hparams.Nframes,
                max_gap=self.hparams.max_gap,
                portion=self.hparams.portion,
                sincos_form=self.hparams.sincos_form,
                train_json=self.hparams.train_json,
                isCache=self.hparams.isCache,
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.unify_type_of_split_params(len(dataset)),
                generator=torch.Generator().manual_seed(42),
            )
            if self.hparams.test_json is not None:
                self.data_test = PviraResultDatasetTest(self.hparams.test_json, sincos_form=self.hparams.sincos_form)

    def unify_type_of_split_params(self, len):
        # raise DeprecationWarning("This method can be removed in the future when upgrade pytroch to 1.13. The function random_split will enable the float fraction input")
        if isinstance(self.hparams.train_val_test_split[0],int):
            return self.hparams.train_val_test_split
        else: # percentage, float
            split = [int(len * x) for x in self.hparams.train_val_test_split[:-1]]
            split.append(len - sum(split))
            return split

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "pvira.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)






















