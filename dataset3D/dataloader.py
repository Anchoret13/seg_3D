import torch 
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
from dataset3D.dataset import ExampleDataset


def get_dataloader(batch_size=3, num_workers=0, shuffle=True, drop_last=False, collate=False):

    dataset = ExampleDataset()
    print("ExampleDataset is made")

    if not collate:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=drop_last
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=drop_last,
            # collate_fn=ME.utils.batch_sparse_collate
            collate_fn=ME.utils.SparseCollation()
        )
    print("ExampleDataset Loader is made")

    return dataloader
