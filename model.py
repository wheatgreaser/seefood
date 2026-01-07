import pandas as pd
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import requests
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from sklearn.preprocessing import LabelEncoder

ds = pd.read_csv('MM-Food-100K/data.csv')
encoder = LabelEncoder()
encoder.fit(ds['dish_name'])

class FoodTrainDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.dataset = (pd.read_csv(csv_file))[:50]
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0] 
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(f"downloaded_images/{index}.jpg")
        row = self.dataset.iloc[index]
        label = encoder.transform([row['dish_name']])[0]
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class FoodTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.dataset = (pd.read_csv(csv_file))[50:100]
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        dataset = self.dataset
        image = io.imread(f"downloaded_images/{50 + index}.jpg")
        row = dataset.iloc[index]
        label = encoder.transform([row['dish_name']])[0]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': sample['label']}

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': sample['label']}

class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': sample['label']}    
    
train_dataset = FoodTrainDataset(csv_file="MM-Food-100K/data.csv", transform=transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ]))


train_dataloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

test_dataset = FoodTestDataset(csv_file="MM-Food-100K/data.csv", transform=transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor()
    ]))

test_dataloader = DataLoader(test_dataset, batch_size=4,
                        shuffle=False, num_workers=0)







    
    

