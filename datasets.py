import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryCircleDataset(Dataset):
    def __init__(self, sample_num=1000):
        self.sample_num = sample_num
        self.radius = 1
        self.radius1 = (0*self.radius, 0.5*self.radius)
        self.radius2 = (0.7*self.radius, self.radius)
        self.radius3 = (1.3*self.radius, 1.6*self.radius)
        self.noise = 0.05

        self.xx, self.yy = self.bi_circle_data_generator()

    def __len__(self):
        return len(self.xx)

    def __getitem__(self, idx):
        return self.xx[idx], self.yy[idx]
        
    def bi_circle_data_generator(self):
        """
        """
        def _make_circle_data(point_num, radius, noise=self.noise):
            r = np.random.uniform(radius[0], radius[1], point_num)
            angles = np.random.uniform(0, 2 * np.pi, point_num)

            xx = np.cos(angles) * r + np.random.uniform(
                -noise*self.radius, 
                noise*self.radius, point_num)
            yy = np.sin(angles) * r + np.random.uniform(
                -noise*self.radius, 
                noise*self.radius, point_num)
            return np.stack([xx, yy], axis=1)

        inner_points = _make_circle_data(
            self.sample_num // 3, 
            self.radius1)

        middle_points = _make_circle_data(
            self.sample_num // 3, 
            self.radius2)

        outter_points = _make_circle_data(
            self.sample_num // 3, 
            self.radius3)

        inner_yy = torch.tensor([0] * len(inner_points))
        middle_yy = torch.tensor([1] * len(outter_points))
        outter_yy = torch.tensor([2] * len(outter_points))

        return torch.tensor(np.concatenate([inner_points, middle_points, outter_points]), dtype=torch.float), \
            torch.cat([inner_yy, middle_yy, outter_yy])

    def scatter(self):
        xx, yy, cc = [], [], []
        for xx_, yy_ in self:
            xx.append(xx_[0])
            yy.append(xx_[1])
            cc.append(yy_)

        plt.scatter(xx, yy, c=cc)


class CircleDataset(Dataset):
    def __init__(self, sample_num=1000):

        self.sample_num = sample_num
        self.width = 0.3
        self.centerx = self.centery = self.center = 2
        self.radius = self.center * 1.5

        self.xx = self.circle_data_generator()

    def scatter(self):
        xx, yy, cc = [], [], []
        for xx_, yy_ in self:
            xx.append(xx_[0])
            yy.append(xx_[1])
            cc.append(yy_)

        plt.scatter(xx, yy, c=cc)

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
        xx = torch.randn(self.sample_num, 2)*self.radius + \
            self.center

        return xx

    def _func_y(self, point):
        if (point[0] - self.centerx)**2 + (point[1] - self.centery) **2 < (self.radius-self.width)**2:
            return 0
        elif (point[0] - self.centerx)**2 + (point[1] - self.centery) **2 < (self.radius+self.width)**2:
            return 1
        else:
            return 2
