import torch
from torch.utils.data import Dataset


class CircleDataset(Dataset):
    def __init__(self):

        self.width = 0.3
        self.centerx = self.centery = self.center = 2
        self.radius = self.center * 1.5

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
        xx = torch.randn(1000, 2)*self.radius + self.center

        return xx

    def _func_y(self, point):
        if (point[0] - self.centerx)**2 + (point[1] - self.centery) **2 < (self.radius-self.width)**2:
            return 0
        elif (point[0] - self.centerx)**2 + (point[1] - self.centery) **2 < (self.radius+self.width)**2:
            return 1
        else:
            return 2
