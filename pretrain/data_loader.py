import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CXRDataset(Dataset):
  def __init__(self, images_path, file_path, augment, data_granularity=0):
    self.img_list = []
    self.augment = augment
    self.data_granularity = data_granularity

    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(images_path, lineItems[0])
          self.img_list.append(imagePath)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    h, w, c = np.array(imageData).shape
    base_crop_size = int(min(h, w)// pow(2,self.data_granularity))
    return self.augment(imageData,base_crop_size)

  def __len__(self):
    return len(self.img_list)