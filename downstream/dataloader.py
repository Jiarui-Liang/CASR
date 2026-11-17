import os
import copy
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split
from albumentations import (
    Compose, HorizontalFlip, Resize, RandomBrightnessContrast, RandomGamma, OneOf, ToFloat,
    ShiftScaleRotate, GridDistortion, ElasticTransform, OpticalDistortion, RandomSizedCrop
)
from albumentations.pytorch import ToTensorV2


def build_transform_segmentation():
  AUGMENTATIONS_TRAIN = Compose([
    OneOf([
        RandomBrightnessContrast(),
        RandomGamma(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(156, 224), height=224, width=224,p=0.25),
    ToFloat(max_value=1)
    ],p=1)

  return AUGMENTATIONS_TRAIN


def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence


# -------------------------------------------------Downstream Segmentation----------------------------------------------
class SegmentationDataset(Dataset):
    """Image/mask folder pairs with identical file names (JSRT-Heart, JSRT-Clavicle, Montgomery)."""
    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3),few_shot=-1):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.img_list= os.listdir(pathImageDirectory)

        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])
            print(f"{few_shot} of total: {len(self.img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)


class multi_ribs_dataset(Dataset):
    def __init__(self,df,transforms,mode,list_label=[],image_root='.'):
        self.df=df
        self.transforms= transforms
        self.mode=mode
        self.list_label=list_label
        self.image_root=image_root
    def __len__(self):
        return len(self.df['img'])
    def __getitem__(self,index):
        img_path=self.df['img'][index]
        img=Image.open(os.path.join(self.image_root, img_path))
        img=img.convert('RGB')
        img=np.asarray(img,dtype=np.float32)
        label0=[]
        for name in self.list_label:
            pts=self.df[name][index]
            label= np.zeros((img.shape[:2]),dtype=np.uint8)
            if pts!='None':
                pts= np.array([[[int(pt['x']),int(pt['y'])]] for pt in pts ])
                label=cv2.fillPoly(label,[pts],255)
            label0.append(label)
        label0=np.stack(label0)
        img=img/255
        label0=label0/255
        label0=label0.transpose((1,2,0))
        dic=self.transforms[self.mode]\
        ( image=img,mask=label0)
        img=dic['image']
        mask=(dic['mask'].permute(2,0,1))

        mask = torch.any(mask > 0.5, dim=0, keepdim=True).float()  # mask: (1, 224, 224)

        return (img, mask)

def VinDr_Rib(data_json_dir,mode='train-val',few_shot=-1,image_root=None):
    # Image paths in the json are relative to the VinDr-RibCXR root, i.e. <root>/data/<split>/<file>.json
    if image_root is None:
        image_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_json_dir))))
    df_data= pd.read_json(data_json_dir)
    list_label=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',\
                         'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
    data_transform= {
        'train-val': Compose([
            Resize(224,224),
            HorizontalFlip(),
            ShiftScaleRotate(rotate_limit=10),
            RandomBrightnessContrast(),
            ToTensorV2()
        ]),
        'test': Compose([
            Resize(224,224),
            ToTensorV2()
        ])
    }
    if mode=='train-val':
        dataset_train_val= multi_ribs_dataset(df_data,data_transform,mode,list_label,image_root)
        total_size = len(dataset_train_val)
        if few_shot <= 0:
            val_size = min(50, max(total_size - 1, 0))
            train_size = total_size - val_size
        else:
            train_size = min(int(few_shot), total_size)
            val_size = total_size - train_size
        torch.manual_seed(42)
        dataset_train, dataset_val = random_split(dataset_train_val, [train_size, val_size])

        val_sample_size = min(50, len(dataset_val))
        val_indices = random.sample(range(len(dataset_val)), val_sample_size)
        dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

        return dataset_train, dataset_val

    elif mode=='test':
        dataset_test = multi_ribs_dataset(df_data,data_transform,mode,list_label,image_root)
        return dataset_test

# ------------------------------------------------Downstream Classification---------------------------------------------
class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, few_shot = -1):
    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)


class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, few_shot = -1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])
            print(f"{few_shot} of total: {len(self.img_list)}")

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel

    def __len__(self):
        return len(self.img_list)
