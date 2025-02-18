import torch
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from typing import Union
from flwr_datasets.partitioner import Partitioner

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Data:
    def __init__(self,
                 batch_size: int,
                 partitioner: Union[int, Partitioner],
                 dataset: str="cifar10",
                 seed: int=42,
                 test_size: float=0.2

                 ) -> None:
        self.dataset = dataset
        self._batch_size = batch_size
        self._partitioner = partitioner
        self._seed = seed
        self.test_size = test_size
        self.fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})

    @staticmethod
    def apply_transforms(batch):
    # Instead of passing transforms to CIFAR10(..., transform=transform)
    # we will use this function to dataset.with_transform(apply_transforms)
    # The transforms object is exactly the same

        pytorch_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    def get_batch_size(self, partition_train_test):

        dataset_sizes = {
        "train": len(partition_train_test["train"]),
        "val": len(partition_train_test["test"]),
        "test": len(self.fds.load_split("test"))
    }

        if self._batch_size == "full" or self._batch_size > dataset_sizes["val"]:
            return dataset_sizes["train"], dataset_sizes["val"], dataset_sizes["test"]
            
        return (self._batch_size, self._batch_size, self._batch_size)


    def load_datasets(self, partition_id: int):
        
        partition = self.fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=self.test_size, seed=self._seed)

        # Create train/val for each partition and wrap it into DataLoader
        partition_train_test = partition_train_test.with_transform(self.apply_transforms)
        
        # Create train, test, val batch sizes
        batch_size_train, batch_size_val, batch_size_test = self.get_batch_size(partition_train_test)
        

        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size_train, shuffle=True
        )
        valloader = DataLoader(partition_train_test["test"], batch_size=batch_size_val)
        testset = self.fds.load_split("test").with_transform(self.apply_transforms)
        testloader = DataLoader(testset, batch_size=batch_size_test)
        return trainloader, valloader, testloader
    
