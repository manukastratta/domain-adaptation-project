import torch
from torchvision import datasets, transforms
import fire
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.functional import to_pil_image
# import matplotlib.pyplot as plt
import colortrans
from torch.multiprocessing import set_start_method

import multiprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Camelyon17Dataset(Dataset):
    def __init__(self, metadata_file, image_dir, transform=None):
        self.metadata = pd.read_csv(image_dir + "/" + metadata_file)
        self.image_dir = image_dir
        self.transform = transform
        self.n_channels = 3
        self.img_size = 96

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.metadata.iloc[idx]['image_path'])
        image = Image.open(image_path)
        # Convert image to a 4-channel image
        image = image.convert('RGBA')
        # Convert image to a 3-channel image by discarding the alpha channel
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        image_np = np.array(image)
        image_tensor = to_tensor(image_np)
        image_tensor = image_tensor.reshape((self.n_channels, self.img_size, self.img_size)) # torch.Size([3, 96, 96])

        #metadata = self.metadata.iloc[idx][['patient', 'node', 'x_coord', 'y_coord', 'slide', 'center']] # ignore split?
        #metadata_tensor = torch.Tensor(metadata.values.astype(np.float32))

        image_path = self.metadata.iloc[idx][['image_path']].item()
        center = self.metadata.iloc[idx][['center']].item()
        metadata = dict({"image_path": image_path, "center": center})

        label = self.metadata.iloc[idx][['tumor']]
        label_tensor = torch.Tensor(label.values.astype(np.int32))

        return image_tensor, label_tensor, metadata

class CelebADataset(Dataset):
    def __init__(self, metadata_file, image_dir, transform=None):
        self.metadata = pd.read_csv(image_dir + "/" + metadata_file)
        self.image_dir = image_dir
        self.transform = transform
        self.n_channels = 3
        self.img_width = 178
        self.img_height = 218

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, "img_align_celeba/" + self.metadata.iloc[idx]['img_filename'])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        image_np = np.array(image) # (3, 218, 178)
        image_tensor = to_tensor(image_np)
        image_tensor = image_tensor.reshape((self.n_channels, self.img_height, self.img_width))

        filename = self.metadata.iloc[idx][['img_filename']].item()
        male = self.metadata.iloc[idx][['Male']].item()
        metadata = dict({"img_filename": filename, "Male": male})
        #metadata_tensor = torch.Tensor(metadata.values)

        label = self.metadata.iloc[idx][['Black_Hair']]
        label_tensor = torch.Tensor(label.values.astype(np.int32))

        return image_tensor, label_tensor, metadata

class FMoWDataset(Dataset):
    def __init__(self, metadata_file, image_dir, transform=None):
        self.metadata = pd.read_csv(image_dir + "/" + metadata_file)
        self.image_dir = image_dir
        self.transform = transform
        self.n_channels = 3
        self.img_width = 224
        self.img_height = 224

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, "images/" + self.metadata.iloc[idx]['new_filename'])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        image_np = np.array(image) # (3, 218, 178)
        image_tensor = to_tensor(image_np)
        image_tensor = image_tensor.reshape((self.n_channels, self.img_height, self.img_width))

        filename = self.metadata.iloc[idx][['new_filename']].item()
        year = self.metadata.iloc[idx][['year']].item()
        metadata = dict({"new_filename": filename, "year": year})
        #metadata_tensor = torch.Tensor(metadata.values)

        label = self.metadata.iloc[idx][['y']]
        label_tensor = torch.Tensor(label.values.astype(np.int32))

        return image_tensor, label_tensor, metadata


# def display_image(image_array):
#     pil_image = to_pil_image(image_array)
#     image = Image.fromarray(np.uint8(pil_image))
#     image.show()

# def display_image(images):
#   images_np = images.numpy()
#   img_plt = images_np.transpose(0,2,3,1)
#   # display 5th image from dataset
#   plt.imshow(img_plt[4])

# def display_image(tensor_img):
#     print(tensor_img.shape)
#     # putting in scale 0-1
#     tensor_img = tensor_img.permute(1, 2, 0)
#     tensor_img = (tensor_img - tensor_img.min()) / (tensor_img.max() - tensor_img.min())
#     plt.imshow(tensor_img )
#     plt.show()

def display_image(tensor_img):
    img = np.array(tensor_img)
    print("img.shape: ", img.shape)
    img = img.transpose(2, 1, 0)
    #img = img.transpose(1, 2, 0)
    print("img.shape: ", img.shape)
    plt.imshow(img)

    import datetime
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"utils/colors/temp/img_{date_string}.png"
    plt.savefig(filename)
    #plt.show()

class LinearHistogramMatching(object):
    def __init__(self):
        # images from unlabeled test set to pick from
        #dataset = "data/camelyon17_unlabeled_v1.0/unlabeled_hospital4.csv"
        # img_names = set(pd.read_csv(dataset)["image_path"])
        # self.img_paths = ["data/camelyon17_unlabeled_v1.0/" + img_name for img_name in img_names]

        # Use test data (treat as unlabeled)
        dataset = "data/camelyon17_v1.0/wilds_splits/metadata_test.csv"
        img_names = set(pd.read_csv(dataset)["image_path"])
        self.img_paths = ["data/camelyon17_v1.0/" + img_name for img_name in img_names]
        
    def __call__(self, img):
        # randomly pick reference image from unlabeled test (hospital 4)
        ref_img_path = np.random.choice(self.img_paths)
        reference = np.array(Image.open(ref_img_path).convert('RGB'))
        x = colortrans.transfer_lhm(np.array(img), reference) # x.shape:  (96, 96, 3)
        return x

def get_camelyon_data_loader(   data_dir,
                                metadata_filename,
                                config,
                                transform=None,
                                num_cpus=multiprocessing.cpu_count()):
    """
    Takes in the meta csv filename, returns dataloader
    Can be used for train, val, or test metadata
    """
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # Teddi values
    # mean = [0.720475903842406, 0.5598634658657207, 0.7148542202373653]
    # std = [0.02169931605678509, 0.025169192291280777, 0.017899670079910467]
    # Manuka values
    # mean = [183.72132071, 142.7651698, 182.28779395]
    # std = [34.25966575, 39.11443915, 28.19893461]
    # Manuka values / 255
    mean = [0.720475767490196, 0.5598634109803922, 0.7148540939215686]
    std = [0.1343516303921569, 0.1533899574509804, 0.11058405729411765]

    if config["target_color_augmentations"]:
        print("Applying target_color_augmentations!")
        transform = transforms.Compose([
                transforms.Resize(96), #(96, 96)
                LinearHistogramMatching(),
                transforms.ToTensor(), # converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range [0.0, 1.0]. 
                #transforms.Normalize(mean, std)
        ])
    else:
        if config["normalization"]:
            print("Normalizing")
            transform = transforms.Compose([
                    transforms.Resize(96), #(96, 96)
                    transforms.ToTensor(), # converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range [0.0, 1.0]. 
                    transforms.Normalize(mean, std)
            ])
        else:
            print("Not normalizing")
            transform = transforms.Compose([
                transforms.Resize(96), #(96, 96)
                transforms.ToTensor(), # converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range [0.0, 1.0]. 
            ])


    dataset = Camelyon17Dataset(metadata_file=metadata_filename, image_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_cpus, persistent_workers=True)

    # # Testing dataloader
    # sample = next(iter(dataloader))
    # image, label, metadata = sample # torch.Size([32, 3, 96, 96]), torch.Size([32, 1]), torch.Size([32, 7])

    # print(image.shape) # torch.Size([32, 3, 96, 96])
    # print(label.shape)
    # single_image = image[4, :, :, :]
    # display_image(single_image)

    return dataloader

def get_celeba_data_loader(data_dir, metadata_filename, batch_size=32, transform=None, num_cpus=multiprocessing.cpu_count()):
    """
    Takes in the meta csv filename, returns dataloader
    Can be used for train, val, or test metadata
    """

    # mean = [129.33003765, 108.17284158, 96.70170372]
    # std = [68.41791949, 63.12533799, 61.96260248]
    # / 255
    # mean = [0.5071766182352941, 0.42420722188235294, 0.37922236752941174]
    # std = [0.268305566627451, 0.24755034505882353, 0.24299059796078432]
    # transform = transforms.Compose([
    #     transforms.Resize((218, 178)), #(96, 96)
    #     transforms.ToTensor(), # converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range [0.0, 1.0]. 
    #     transforms.Normalize(mean, std)
    # ])
    transform = None # Experiments showed that no normalization is best

    dataset = CelebADataset(metadata_file=metadata_filename, image_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpus, persistent_workers=True)

    # # Testing dataloader
    # sample = next(iter(dataloader))
    # image, label, metadata = sample # torch.Size([32, 3, 96, 96]), torch.Size([32, 1]), torch.Size([32, 7])

    # print(image.shape)  # torch.Size([32, 3, 218, 178])
    # print(label.shape)  # torch.Size([32, 1])
    # single_image = image[4, :, :, :]
    # display_image(single_image)

    return dataloader


def get_fmow_data_loader(data_dir, metadata_filename, batch_size=32, transform=None, num_cpus=multiprocessing.cpu_count()):
    """
    Takes in the meta csv filename, returns dataloader
    Can be used for train, val, or test metadata
    """
    transform = None # Experiments showed that no normalization is best
    dataset = FMoWDataset(metadata_file=metadata_filename, image_dir=data_dir, transform=transform)
    # TODO change back to shuffle True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpus, persistent_workers=True)

    # # Testing dataloader
    # sample = next(iter(dataloader))
    # image, label, metadata = sample # torch.Size([32, 3, 96, 96]), torch.Size([32, 1]), torch.Size([32, 7])

    # print(image.shape)  # torch.Size([32, 3, 218, 178])
    # print(label.shape)  # torch.Size([32, 1])
    # single_image = image[2, :, :, :]
    # display_image(single_image)

    return dataloader


if __name__ == '__main__':
    fire.Fire()