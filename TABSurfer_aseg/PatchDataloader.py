import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import class_weight
import ipdb

class PatchDataset(Dataset):
    def __init__(self, input_img, target_img): # cropped, masked, and vectorized
        self.patch_size = 96
        self.image1 = input_img
        self.image2 = target_img
        #assert np.array_equal(self.image1.shape, self.image2.shape), "Arrays do not have the same shape"
        self.x_size, self.y_size, self.z_size = self.image1.shape #160, 176, 208
        self.x_step_size = (self.x_size - self.patch_size) // 2
        self.y_step_size = (self.y_size - self.patch_size) // 2
        self.z_step_size = (self.z_size - self.patch_size) // 2

        #self.class_weights=np.array(class_weight.compute_class_weight('balanced',classes = np.unique(target_img),y = target_img.ravel()))

        #self.weights_image = self.class_weights[self.image2]

    def __len__(self):
        return 27

    def __getitem__(self, index): # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
        x = (index // 9) * self.x_step_size
        y = (index // 3 % 3) * self.y_step_size
        z = (index % 3) * self.z_step_size
        # 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2
        # 0 0 0 0 0 1 1 1 1 2 2 2 2 0 0 0 0 1 1 1 1 2 2 2 2 0 0 0 0 1 1 1 1 2 2 2 2
        # 0 1 2 3 4 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
        #ipdb.set_trace()
        patch1 = self.image1[x : x + self.patch_size, y : y + self.patch_size, z : z + self.patch_size]
        patch2 = self.image2[x : x + self.patch_size, y : y + self.patch_size, z : z + self.patch_size]
        #weights = self.weights_image[x : x + self.patch_size, y : y + self.patch_size, z : z + self.patch_size]
        return patch1, patch2#, weights
    
def get_patch_dataloader(conformed_scan, segmented_scan, batch_size = 2, shuffle = False, num_workers = 0, pin_memory=True):
    patch_dataset = PatchDataset(conformed_scan, segmented_scan)
    data_loader = DataLoader(dataset=patch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader
