"""
BloodMNIST data utilities for federated learning.

Handles data loading, preprocessing, and partitioning across clients.
Supports both IID and non-IID data distributions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from typing import List, Tuple, Optional
import medmnist
from medmnist import BloodMNIST


class BloodMNISTWrapper(Dataset):
    """Wrapper for BloodMNIST with custom transforms."""

    def __init__(
        self,
        split: str = "train",
        img_size: int = 224,
        download: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            split: One of "train", "val", "test"
            img_size: Target image size
            download: Whether to download dataset if not present
            transform: Optional custom transform
        """
        # Load BloodMNIST with appropriate size
        # MedMNIST supports: 28, 64, 128, 224
        # Download the dataset if not present
        size = img_size if img_size in [28, 64, 128, 224] else 224

        self.dataset = BloodMNIST(
            split=split,
            download=download,
            size=size,
            as_rgb=True, 
        )
        # Apply default transforms to data if no custom transform is provided
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform(img_size, split == "train")

    def _default_transform(self, img_size: int, is_train: bool) -> transforms.Compose:
        """Create default transforms for training/evaluation."""
        # If training, apply random horizontal flip, rotation, color jitter, and normalization
        if is_train:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            # If evaluation, apply normalization only
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the image and label from the dataset
        img, label = self.dataset[idx]
        # Apply the transform to the image
        img = self.transform(img)
        # BloodMNIST labels are shape (1,), squeeze to scalar
        label = torch.tensor(label.squeeze(), dtype=torch.long)
        return img, label


def partition_data_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition data IID (independent and identically distributed).

    Each client gets a random subset of roughly equal size.

    Args:
        dataset: The full dataset
        num_clients: Number of clients to partition for
        seed: Random seed for reproducibility

    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)

    indices = np.random.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)

    return [split.tolist() for split in splits]


def partition_data_non_iid(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition data non-IID using Dirichlet distribution.

    Lower alpha = more heterogeneous distribution (more non-IID).
    Higher alpha = more homogeneous distribution (closer to IID).

    Args:
        dataset: The full dataset
        num_clients: Number of clients to partition for
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility

    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)

    # Get labels for all samples
    labels = np.array([dataset[i][1].item() if isinstance(dataset[i][1], torch.Tensor)
                       else dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    # Create client data indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples according to Dirichlet distribution
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Calculate number of samples per client for this class
        proportions = (proportions * len(class_indices)).astype(int)

        # Handle rounding errors
        proportions[-1] = len(class_indices) - proportions[:-1].sum()

        # Assign indices to clients
        start = 0
        for client_idx, num_samples in enumerate(proportions):
            client_indices[client_idx].extend(
                class_indices[start:start + num_samples].tolist()
            )
            start += num_samples

    # Shuffle each client's data
    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices


class FederatedDataset:
    """
    Manages federated dataset partitioning and access.

    Example:
        fed_data = FederatedDataset(num_clients=5, partition="non_iid")
        train_loader = fed_data.get_client_dataloader(client_id=0, batch_size=32)
    """

    def __init__(
        self,
        num_clients: int = 5,
        partition: str = "iid",  # "iid", "non_iid"
        alpha: float = 0.5,  # For non_iid
        img_size: int = 28,
        seed: int = 42,
    ):
        """
        Args:
            num_clients: Number of clients in federation
            partition: Partitioning strategy
            alpha: Dirichlet alpha for non_iid (lower = more heterogeneous)
            img_size: Image size for transforms
            seed: Random seed for reproducibility
        """
        # Initialize the FederatedDataset
        self.num_clients = num_clients
        self.partition = partition
        self.img_size = img_size

        # Load datasets (suppress medmnist download messages)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.train_dataset = BloodMNISTWrapper(split="train", img_size=img_size)
            self.val_dataset = BloodMNISTWrapper(split="val", img_size=img_size)
            self.test_dataset = BloodMNISTWrapper(split="test", img_size=img_size)
        finally:
            sys.stdout = old_stdout

        print(f"Dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
        # Partition the data into clients by iid or non_iid
        if partition == "iid":
            self.client_indices = partition_data_iid(
                self.train_dataset, num_clients, seed
            )
        elif partition == "non_iid":
            self.client_indices = partition_data_non_iid(
                self.train_dataset, num_clients, alpha, seed
            )
        else:
            raise ValueError(f"Unknown partition: {partition}")

        # Print partition statistics
        self._print_partition_stats()

    def _print_partition_stats(self):
        """Print statistics about the data partition."""
        samples_per_client = [len(indices) for indices in self.client_indices]
        print(f"Partition: {self.partition}, samples per client: {samples_per_client}")

    def get_client_dataloader(
        self,
        client_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get training DataLoader for a specific client."""
        # Check if the client id is valid
        if client_id >= self.num_clients:
            raise ValueError(f"Client {client_id} does not exist")
        # Get the indices for the client
        indices = self.client_indices[client_id]
        # Create a subset of the dataset for the client
        subset = Subset(self.train_dataset, indices)
        # Create a dataloader for the client
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # Avoid MPS warnings
        )

    def get_val_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get validation DataLoader (shared across all clients)."""
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    def get_test_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get test DataLoader (for final evaluation)."""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    def get_client_dataset_size(self, client_id: int) -> int:
        """Get the number of samples for a specific client."""
        return len(self.client_indices[client_id])
