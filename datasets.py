import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import os
import torch
from torch.utils.data import Dataset, Sampler
import sys
sys.path.append("/home/anjum/PycharmProjects/kaggle")
# sys.path.append("/home/anjum/rsna_code")  # GCP
from rsna_intracranial_hemorrhage_detection.data_prep import linear_windowing, sigmoid_windowing
ImageFile.LOAD_TRUNCATED_IMAGES = True


data_path = "/mnt/storage_dimm2/kaggle_data/rsna-intracranial-hemorrhage-detection/"
# data_path = "/home/anjum/rsna_data/"  # GCP


class ICHDataset(Dataset):
    def __init__(self, dataset, phase=1, image_filter=None, transforms=None, image_folder=None, png=True):
        df_paths = {
            "train": os.path.join(data_path, "stage_1_train.csv"),
            "test1": os.path.join(data_path, "stage_1_sample_submission.csv"),
            "test2": os.path.join(data_path, "stage_2_sample_submission.csv")
        }

        self.png = png

        if self.png:
            image_dirs = {
                "train": os.path.join(data_path, "png", "train", image_folder),
                "test1": os.path.join(data_path, "png", "test_stage_1", image_folder),
                "test2": os.path.join(data_path, "png", "test_stage_2", image_folder)
            }
        else:
            image_dirs = {
                "train": os.path.join(data_path, "npy", "train", image_folder),
                "test1": os.path.join(data_path, "npy", "test_stage_1", image_folder),
                "test2": os.path.join(data_path, "npy", "test_stage_2", image_folder)
            }

        self.dataset = dataset
        self.phase = phase
        self.transforms = transforms
        self.image_dir = image_dirs[dataset]

        self.df = pd.read_csv(df_paths[dataset]).drop_duplicates()
        self.df['ImageID'] = self.df['ID'].str.slice(stop=12)
        self.df['Diagnosis'] = self.df['ID'].str.slice(start=13)
        self.df_pivot = self.df.pivot(index="ImageID", columns="Diagnosis", values="Label")

        if image_filter is not None:
            self.df_pivot = self.df_pivot.loc[image_filter]

        if self.phase == 0:
            self.labels = self.df_pivot.values
        elif self.phase == 1:
            self.labels = self.df_pivot["any"].values.reshape(-1, 1)
        else:
            self.labels = self.df_pivot[["epidural", "intraparenchymal",
                                         "intraventricular", "subarachnoid", "subdural"]].values

        self.image_ids = self.df_pivot.index.values
        self.class_weights = np.mean(self.labels, axis=0)

    def load_image(self, image_name):
        # window_width, window_length = 80, 40  # Brain window
        window_width, window_length = 200, 80  # Subdural window
        # window_width, window_length = 130, 50  # Subdural window

        if self.png:
            img = np.array(Image.open(os.path.join(self.image_dir, image_name+".png")).convert("RGB"))
            return linear_windowing(img, window_width, window_length)
            # return sigmoid_windowing(img, window_width, window_length)
        else:
            img = np.load(os.path.join(self.image_dir, image_name+".npy"))

            # PIL doesn't work with 16-bit RGB images :(
            # Should be ok though since the useful HU interval is between 0-255
            if img.shape[0] == 0 or img.shape[1] == 0:
                return np.zeros(shape=(512, 512, 3), dtype=np.uint8)
            else:
                # return np.clip(img, 0, 255).astype(np.uint8)  # Use this with the Windowing module
                return linear_windowing(img, window_width, window_length)
                # return sigmoid_windowing(img, window_width, window_length)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = self.load_image(img_id)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.dataset == "train":
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return img, torch.tensor([0], dtype=torch.float32)

    def __len__(self):
        return len(self.image_ids)


class BalancedRandomSampler(Sampler):
    def __init__(self, data_source):
        """
        Balances the negative and positive samples. All of the positive samples are used, but a random subset of
        the negative samples are used to create a 50:50 dataset
        :param data_source: An ICHDataset
        """
        super().__init__(data_source)
        self.labels = data_source.labels
        self.ids_pos = np.where(self.labels[:, 0] == 1)[0]
        self.ids_neg = np.where(self.labels[:, 0] == 0)[0]

    def __iter__(self):
        ids_neg_sampled = np.random.choice(self.ids_neg, self.ids_pos.shape[0], replace=False)
        ids = np.concatenate([self.ids_pos, ids_neg_sampled])
        np.random.shuffle(ids)
        return iter(ids)

    def __len__(self):
        return self.ids_pos.shape[0] * 2
