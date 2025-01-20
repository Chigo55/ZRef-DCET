import os
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class TrainDataLoader(Dataset):
    """
    Custom Dataset class for loading training images.

    Args:
        data_path (str): Path to the directory containing training images.
        transform (transforms.Compose): Transformations to be applied to each image.

    Attributes:
        data_path (Path): Path object pointing to the training data directory.
        image_paths (list): List of file paths for all images in the directory.
        transform (transforms.Compose): Transformations to be applied to the images.
    """

    def __init__(self, data_path: str, transform: transforms.Compose) -> None:
        super().__init__()
        self.data_path = Path(data_path)

        # List all image files in the directory
        self.image_paths = [self.data_path / fname for fname in self.data_path.iterdir() if fname.is_file()]
        self.transform = transform

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image by index, applies transformations, and returns the processed image.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tensor: Transformed image.
        """
        img_path = self.image_paths[idx]  # Get the path for the given index
        image = Image.open(img_path)  # Open the image
        if self.transform:
            image = self.transform(image)  # Apply transformations
        return image  # Return transformed image


class TestDataLoader(Dataset):
    """
    Custom Dataset class for loading testing images.

    Args:
        data_path (str): Path to the directory containing testing images.
        transform (transforms.Compose): Transformations to be applied to each image.

    Attributes:
        data_path (Path): Path object pointing to the testing data directory.
        image_paths (list): List of file paths for all images in the directory.
        transform (transforms.Compose): Transformations to be applied to the images.
    """

    def __init__(self, data_path: str, transform: transforms.Compose) -> None:
        super().__init__()
        self.data_path = Path(data_path)

        # List all image files in the directory
        self.image_paths = [self.data_path / fname for fname in self.data_path.iterdir() if fname.is_file()]
        self.transform = transform

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image by index, applies transformations, and returns the processed image.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tensor: Transformed image with dimensions permuted to (C, H, W).
        """
        img_path = self.image_paths[idx]  # Get the path for the given index
        image = Image.open(img_path)  # Open the image
        if self.transform:
            image = self.transform(image)  # Apply transformations
        return image.permute(2, 0, 1)  # Change dimensions to (C, H, W)


def create_dataloaders(
    train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int = os.cpu_count()
):
    """
    Creates DataLoaders for training and testing datasets.

    Args:
        train_dir (str): Path to the directory containing training data.
        test_dir (str): Path to the directory containing testing data.
        transform (transforms.Compose): Transformations to apply to the data.
        batch_size (int): Number of samples per batch in each DataLoader.
        num_workers (int): Number of worker processes for data loading (default: number of CPU cores).

    Returns:
        tuple: A tuple containing:
            - train_dataloader (DataLoader): DataLoader for the training dataset.
            - test_dataloader (DataLoader): DataLoader for the testing dataset.
    """
    # Initialize custom dataset objects for training and testing
    train_data = TrainDataLoader(data_path=train_dir, transform=transform)
    test_data = TestDataLoader(data_path=test_dir, transform=transform)

    # Create DataLoaders for training and testing datasets
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, test_dataloader
