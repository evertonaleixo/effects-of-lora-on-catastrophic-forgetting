from avalanche.benchmarks.classic import SplitCUB200, SplitCIFAR100
from torchvision import transforms

from pydantic import BaseModel
from typing import List

class DatasetMetadata(BaseModel):
    total_number_classes: int
    n_split_experiences: int


class ContinuousLearninDataset(BaseModel):
    name: str = None
    train_ds: List = []
    test_ds: List = []
    masks: List = []
    classes_in_task: List = []
    metadata: DatasetMetadata = None


def get_dataset_metadata(name: str) -> DatasetMetadata:
    assert name in ('cub-200', 'cifar-100')

    metadatas = {
        'cub-200': DatasetMetadata(
            total_number_classes=200,
            n_split_experiences=5
        )
    }

    return metadatas[name]

def get_dataset(name: str, seed:int=1234) -> ContinuousLearninDataset:
    assert name in ('cub-200', 'cifar-100')

    metadata = get_dataset_metadata(name=name)

    if name == 'cub-200':
        split_ds = SplitCUB200(
            n_experiences=metadata.n_split_experiences,
            seed=seed,
            return_task_id=True,
            train_transform=transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            eval_transform=transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        )
    elif name == 'cifar-100':
        split_ds = SplitCIFAR100(
            n_experiences=metadata.n_split_experiences,
            seed=seed,
            return_task_id=True
        )
    
    train_stream = split_ds.train_stream
    test_stream = split_ds.test_stream

    train_ds = []
    test_ds = []
    masks = []
    classes_in_task = []

    for experience in train_stream:
        masks.append([i for i in range(metadata.total_number_classes) if i not in experience.classes_in_this_experience])
        classes_in_task.append(experience.classes_in_this_experience)
        
        current_training_set = experience.dataset
        train_ds.append(current_training_set)
        
        current_test_set = test_stream[experience.current_experience].dataset
        test_ds.append(current_test_set)

    return ContinuousLearninDataset(
        name=name,
        train_ds=train_ds,
        test_ds=test_ds,
        masks=masks,
        classes_in_task=classes_in_task,
        metadata=metadata,
    )
    

