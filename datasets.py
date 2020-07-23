import torch
from torch.utils.data import Dataset


class CircleDataset(Dataset):
    def __init__(self):
        self.xx = self.circle_data_generator()

    def __len__(self):
        return len(self.xx)

    def __getitem__(self, idx):
        item_x = self.xx[idx]
        return item_x, self._func_y(item_x)

    def circle_data_generator(self):
        """
            0 - in circle
            1 - on circle
            2 - out circle
            radius = 0.8
        """
        width = 0.3
        centerx = centery = center = 2
        radius = center * 1.5
        xx = torch.randn(1000, 2)*radius + center

        return xx

    def _func_y(self, point):
        if (point[0] - centerx)**2 + (point[1] - centery) **2 < (radius-width)**2:
            return 0
        elif (point[0] - centerx)**2 + (point[1] - centery) **2 < (radius+width)**2:
            return 1
        else:
            return 2
