import torch
import torch.nn.functional
import torch.utils.data
import torchvision

import numpy

import os
import pathlib

import PIL.Image



class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: pathlib.Path, device: torch.device) -> None:
        """
        Args:
            root_dir (pathlib.Path): Directory with all the images.
        """
        super(ImageDataset, self).__init__()
        
        data_path: pathlib.Path = data_path
            
        input_path = data_path / 'input'
        ground_truth_path = data_path / 'ground_truth'

        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []


        zipped = zip(input_path.iterdir(), ground_truth_path.iterdir())
        for img_input_path, img_ground_truth_path in zipped:

            #filename = img_input_path.name #filename with extension file
            filename = img_input_path.stem #filename without extension file
            
            # Load imgs
            img_input = torch.tensor(numpy.load(img_input_path))
            img_ground_truth = torch.tensor(numpy.load(img_ground_truth_path))
            
            # Move on datas device
            img_input = img_input.to(device, dtype=torch.float)
            img_ground_truth = img_ground_truth.to(device, dtype=torch.float)
            
            self.items.append((img_input, img_ground_truth, filename))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.items[index]


def split_dataset(dataset: torch.utils.data.Dataset, train_size: float) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    n = len(dataset)
    train_n = int(train_size*n)
    test_n = n-train_n
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_n, test_n])
    return train_dataset, test_dataset


def get_batch_with_variable_size_image(batch):

    imgs_input = []
    imgs_ground_truth = []
    imgs_filename = []

    for elem in batch:
        imgs_input.append(elem[0])
        imgs_ground_truth.append(elem[1])
        imgs_filename.append(elem[2])

   
    # Your custom processing here
    return imgs_input, imgs_ground_truth, imgs_filename

def from_config(
    config: dict
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    # Dataset params
    dataset_path = pathlib.Path(config['dataset']['path'])
    datas_device = config['dataset']['device']
    batch_size = config['dataset']['params']['batch_size']
    train_size = config['dataset']['params']['train_size']

    dataset_full = ImageDataset(dataset_path, datas_device)
    dataset_train, dataset_validation = split_dataset(dataset_full, train_size=train_size)


    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        collate_fn = get_batch_with_variable_size_image,
        shuffle=True
    )


    dataloader_validation= torch.utils.data.DataLoader(
        dataset_validation, 
        batch_size=batch_size,
        collate_fn = get_batch_with_variable_size_image,
        shuffle=True,
    )

    return dataloader_train, dataloader_validation