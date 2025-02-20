from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from typing import Union
from flwr_datasets.partitioner import Partitioner
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Data:
    def __init__(self,
                 batch_size: int,
                 partitioner: Union[int, Partitioner],
                 dataset: str = "cifar10",
                 seed: int = 42,
                 val_size: float = 0.2,
                 val_test_batch_size: int=64) -> None:
        
        self.dataset = dataset
        self._batch_size = batch_size  # The batch size for training
        self._val_test_batch_size = val_test_batch_size  # The batch size for validation and testing
        self._partitioner = partitioner
        self._seed = seed
        self._val_size = val_size
        self.fds = FederatedDataset(dataset=self.dataset,
                                    partitioners={"train": self._partitioner},
                                    seed=self._seed)
        
        # Create test loader before partitioning the data, so that the test data is the same for all partitions
        self.create_test_loader()

    @staticmethod
    def apply_transforms(batch):
        """Apply PyTorch transforms to the dataset."""
        pytorch_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    def get_batch_size(self, partition_train_test):
        """Determine the batch size for training, validation, and testing."""
        trainset_size = len(partition_train_test["train"])

        if self._batch_size == "full" or self._batch_size > trainset_size:
            return trainset_size, self._val_test_batch_size  # Full train batch, fixed val/test batch
        else:
            return self._batch_size, self._val_test_batch_size  # Custom train batch, fixed val/test batch

    def create_test_loader(self) -> None:
        """Create test data loader before partitioning."""
        testset = self.fds.load_split("test").with_transform(self.apply_transforms)
        self.testloader = DataLoader(testset, batch_size=self._val_test_batch_size)

    def load_datasets(self, partition_id: int):
        """Load training and validation datasets for a given partition."""
        partition = self.fds.load_partition(partition_id)
        
        # Split into train (80%) and validation (20%)
        partition_train_test = partition.train_test_split(test_size=self._val_size, seed=self._seed)
        
        # Apply transformations
        partition_train_test = partition_train_test.with_transform(self.apply_transforms)
        
        # Get batch sizes
        batch_size_train, batch_size_val = self.get_batch_size(partition_train_test)
        
        # Create DataLoaders
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size_train, shuffle=True)
        valloader = DataLoader(partition_train_test["test"], batch_size=batch_size_val)

        return trainloader, valloader, self.testloader
