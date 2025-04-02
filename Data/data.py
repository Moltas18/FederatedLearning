# Torch imports
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Flower imports
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner

# Other imports
import numpy as np
from typing import Union, Sequence
from datasets import DatasetDict

class Data:

    def __init__(self,
                 batch_size: Union[str, int, np.integer, torch.Tensor],
                 partitioner: Union[int, Partitioner],
                 partition_size: Union[int, np.integer, torch.Tensor]=None,
                 dataset: str = "AnnantJain/aircraft",
                 seed: Union[int, np.integer] = 42,
                 include_test_set = False,
                 val_size: float = 0.5,
                 val_test_batch_size: Union[int, np.integer, torch.Tensor]=64,
                 normalization_means: Sequence[float] =  (0.4914, 0.4822, 0.4465),
                 normalization_stds: Sequence[float] = (0.247, 0.243, 0.261)
                 ) -> None:
        
        self.dataset = dataset
        self._batch_size = batch_size  # The batch size for training
        self._val_test_batch_size = int(val_test_batch_size)  # The batch size for validation and testing
        self._partitioner = partitioner 
        self._partition_size = partition_size
        self._seed = int(seed)
        self._val_size = val_size
        self._include_test_set = include_test_set

        assert len(normalization_means) == 3 and len(normalization_stds) == 3, \
            "Normalization parameters must have exactly 3 values (for RGB channels)."
        
        self.normalization_means = normalization_means
        self.normalization_stds = normalization_stds

        

        # Validate partitioner type
        if not isinstance(self._partitioner, (int, np.integer, torch.Tensor, Partitioner)):
            print(type(self._partitioner))
            raise ValueError("partitioner must be either an int, a numpy integer, a torch tensor, or a Partitioner instance")
        
        # Create the apply_transformation function
        # self.apply_transforms = self._create_apply_transforms()
        
        # Create the fds 
        self._create_fds()

        # Create test loader before partitioning the data, so that the test data is the same for all partitions
        self._create_test_loader()

    def _create_fds(self):
        
        # If the partitioner is an int, the dataset will be scaled down to only include many enough images.
        if isinstance(self._partitioner, (int, np.integer, torch.Tensor)):
        
            if not isinstance(self._partition_size, (int, np.integer, torch.Tensor)):
                raise ValueError("partition_size must be an int if partitioner is an int")
            
            else:
                self._partition_size = int(self._partition_size)
                self._partitioner = int(self._partitioner)
                total_data_size = self._partition_size * self._partitioner


                self.fds = FederatedDataset(dataset=self.dataset,
                                       partitioners={"train": self._partitioner},
                                       seed=self._seed,
                                       preprocessor=lambda d: self.trim_dataset(d, total_data_size),
                                       shuffle=False)

        # If partitioner is a Partitioner, this will be passed forward to the FederatedDataset   
        else:
            self.fds = FederatedDataset(dataset=self.dataset,
                                        partitioners={"train": self._partitioner},
                                        seed=self._seed,
                                        shuffle=False)

    @staticmethod
    def trim_dataset(dataset_dict: DatasetDict, n: int) -> DatasetDict:
        """Trims each dataset in DatasetDict to the first N samples."""
        return DatasetDict({k: v.select(range(min(n, len(v)))) for k, v in dataset_dict.items()})
    
    @staticmethod
    def apply_transforms(batch):
        """Apply PyTorch transforms to the dataset."""
        pytorch_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        # Convert images to tensors
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]

        # Map labels to numerical values and convert to tensors
        label_to_index = {"aircraft": 0}  # Add mappings for all possible labels
        batch["label"] = [torch.tensor(label_to_index[label]) for label in batch["label"]]
        return batch

    def _get_batch_size(self, partition_train_test):
        """Determine the batch size for training, validation, and testing."""
        trainset_size = len(partition_train_test["train"])

        if self._batch_size == "full" or self._batch_size > trainset_size:
            return trainset_size, self._val_test_batch_size  # Full train batch, fixed val/test batch
        else:
            return int(self._batch_size), self._val_test_batch_size  # Custom train batch, fixed val/test batch

    def _create_test_loader(self) -> None:
        """Create test data loader before partitioning."""
        if not self._include_test_set:
            self.testloader = DataLoader(EmptyDataset(), batch_size=self._val_test_batch_size)
            return

        testset = self.fds.load_split("test").with_transform(self.apply_transforms)
        self.testloader = DataLoader(testset, batch_size=self._val_test_batch_size)

    def load_datasets(self, partition_id: int):
        """Load training and validation datasets for a given partition."""
        partition = self.fds.load_partition(partition_id)
        
        # Split into train (100(1-test_size)%) and validation (test_size*100%)
        partition_train_test = partition.train_test_split(test_size=self._val_size, seed=self._seed)
        
        # Apply transformations
        partition_train_test = partition_train_test.with_transform(self.apply_transforms)
        
        # Get batch sizes
        batch_size_train, batch_size_val = self._get_batch_size(partition_train_test)
        
        # Create DataLoaders
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size_train, shuffle=False)
        valloader = DataLoader(partition_train_test["test"], batch_size=batch_size_val, shuffle=False)

        return trainloader, valloader, self.testloader

    def get_config_dict(self):
        """Return a dictionary with all parameters needed to recreate the dataset."""
        config_dict = {
            "dataset": self.dataset,
            "batch_size": self._batch_size,
            "val_test_batch_size": self._val_test_batch_size,
            "partitioner": self._partitioner if isinstance(self._partitioner, int) else self._partitioner.__class__.__name__,
            "partition_size": self._partition_size,
            "seed": self._seed,
            "val_size": self._val_size,
            "include_test_set": self._include_test_set,
            "normalization_means": list(self.normalization_means),
            "normalization_stds": list(self.normalization_stds)
        }
        return config_dict
    
class EmptyDataset(Dataset):
    """An empty dataset returning no samples."""
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")

if __name__ == '__main__':
    import os
    import sys

    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

    from src.utils import denormalize
    d = Data(
        batch_size=1,
        partitioner= 10,
        partition_size = 2,
        val_size=0.5
    )

    trainloader, valloader, testloader = d.load_datasets(partition_id=0)

    print(f"Number of samples in the training dataset: {len(trainloader.dataset)}")
    print(f"Number of samples in the validation dataset: {len(valloader.dataset)}")

    x = next(iter(trainloader))
    print(x)
    # print(x['img'])
    # x_ = denormalize(x['img'],(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    # print(x_)