# datasets.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class cancer_dataset(Dataset):
    def __init__(self, data_dir, transform, dataset_type=None):
        path2data = os.path.join(data_dir, "data_sample/data_sample")
        filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        path2labels = os.path.join(data_dir, "labels.csv")
        labels_df = pd.read_csv(path2labels)
        labels_df.set_index("id", inplace=True)

        if dataset_type == "train":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames][0:3608]
            self.full_filenames = [os.path.join(path2data, f) for f in filenames][0:2608]
        elif dataset_type == "val":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames][3608:3648]
            self.full_filenames = [os.path.join(path2data, f) for f in filenames][3508:3648]
        elif dataset_type == "test":
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames][3648:-1]
            self.full_filenames = [os.path.join(path2data, f) for f in filenames][3648:-1]
        else:
            self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]

        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.full_filenames[idx])
        img = self.transform(img)
        return img, self.labels[idx]
