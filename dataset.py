from typing import Union

import h5py
import torch
from torch.utils.data import Dataset

class MyEEGDataset(Dataset):
    def __init__(self, x: Union[torch.Tensor, tuple[torch.Tensor, ...]], y: torch.Tensor):
        if isinstance(x, torch.Tensor):
            assert x.size(0) == len(y), "数据和标签样本数量不匹配"
            self._x_type = torch.Tensor
        elif isinstance(x, tuple):
            assert all(xi.size(0) == len(y) for xi in x), "数据和标签样本数量不匹配"
            self._x_type = tuple
        else:
            raise TypeError("x must be a torch.Tensor or a tuple of torch.Tensor")
        
        self.x = x
        self.y = y

    def __getitem__(self, index) -> tuple[Union[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]:
        if self._x_type is tuple:
            x = tuple(xi[index] for xi in self.x)
        else:
            x = self.x[index]
        return x, self.y[index]
    
    def __len__(self):
        return len(self.y)


class Hdf5Dataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'r')

    def __del__(self):
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")


class MySeedDataset(Hdf5Dataset):
    def __init__(self, file_path: str, transform=None):
        super().__init__(file_path)
        self.transform = transform

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
