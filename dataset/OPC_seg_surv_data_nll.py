import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
from skimage import measure
import pandas as pd
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
from torch.utils.data.sampler import BatchSampler, Sampler
import torch.distributed as dist
import math
import SimpleITK as sitk


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = True


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)
        print("weights:",weights)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def cc2weight(cc, w_min: float = 1., w_max: float = 2e5):
    weight = torch.zeros_like(cc, dtype=torch.float32)
    cc_items = torch.unique(cc)
    K = len(cc_items) - 1
    N = torch.prod(torch.tensor(cc.shape))
    for i in cc_items:
        weight[cc == i] = N / ((K + 1) * torch.sum(cc == i))
    return torch.clip(weight, w_min, w_max)

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)
        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        seg = sample['seg']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            seg = np.flip(seg, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            seg = np.flip(seg, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            seg = np.flip(seg, 2)

        return {'image': image, 'seg': seg, 'ids':sample['ids'],'Status':sample['Status'], 'censorship':sample['censorship'],'survival_months':sample['survival_months'],'label':sample['label']}

from scipy.ndimage import zoom
class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        # image = image[61: 61 + 128, 61: 61 + 128, 11: 11 + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        seg = sample['seg']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': seg, 'ids':sample['ids'],'label':sample['label']}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        #image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = image[np.newaxis,:]
        image = np.ascontiguousarray(image,np.float32)

        # factor = 0.1
        # scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        # shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        # image = image*scale_factor+shift_factor


        seg = sample['seg']
        seg = seg[np.newaxis,:]
        seg = np.ascontiguousarray(seg,np.uint8)

        #cc = measure.label(seg, connectivity=3, background=0)

        image = torch.from_numpy(image).float()
        seg = torch.from_numpy(seg).long()
        #cc = torch.from_numpy(cc).long()
        #weight = cc2weight(cc)
        weight = 0
        return {'image': image, 'seg': seg,'weight':weight, 'ids':sample['ids'],'Status':sample['Status'], 'censorship':sample['censorship'],'survival_months':sample['survival_months'],'label':sample['label']}

class ToTensor_test(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        #image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = image[np.newaxis,:]
        image = np.ascontiguousarray(image,np.float32)


        seg = sample['seg']
        seg = seg[np.newaxis,:]
        seg = np.ascontiguousarray(seg,np.uint8)

        #cc = measure.label(seg, connectivity=3, background=0)

        image = torch.from_numpy(image).float()
        seg = torch.from_numpy(seg).long()
        #cc = torch.from_numpy(cc).long()
        #weight = cc2weight(cc)
        weight = 0
        return {'image': image, 'seg': seg,'weight':weight, 'ids':sample['ids'],'Status':sample['Status'], 'censorship':sample['censorship'],'survival_months':sample['survival_months'],'label':sample['label']}

class Augmentation(object):
    """Augmentation for the training data.

   :array: A numpy array of size [c, x, y, z]
   :returns: augmented image and the corresponding mask

   """
    def __call__(self,sample):
        array = sample['image']
        mask = sample['seg']
        print('array的shape：',array.shape)
        array = array.transpose(3, 0, 1, 2)
        # normalize image to range [0, 1], then apply this transform
        patch_size = np.asarray(array.shape)[1:]
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))

        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None, ...]
        mask = mask[None,None, ...]
        r_range = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.

        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=3,
            p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        mask = mask[0][0]
        image = augmented[0]
        image = image.transpose(1,2,3,0)
        return {'image': image,'seg':mask, 'ids':sample['ids'],'Status':sample['Status'], 'censorship':sample['censorship'],'survival_months':sample['survival_months']}


def transform(sample):
    trans = transforms.Compose([
        #Pad(),
        #Random_rotate(),  # time-consuming
        # Random_Crop(),
        Random_Flip(),
        #Random_intencity_shift(),
        # Augmentation(),
        ToTensor()
    ])
    return trans(sample)



def transform_valid(sample):
    trans = transforms.Compose([
        #Pad(),
        # MaxMinNormalization(),
        # Random_Crop(),
        ToTensor_test()
    ])

    return trans(sample)

def transform_test(sample):
    trans = transforms.Compose([
        #Pad(),
        # MaxMinNormalization(),
        # Random_Crop(),
        ToTensor()
    ])

    return trans(sample)


class OPCData(Dataset):
    def __init__(self, list_file, root='', mode='train'):        
        self.data = pd.read_csv(list_file)
        self.mode = mode

    def __getitem__(self, item):
        image = sitk.ReadImage(list(self.data.iloc[[item]]['CT'])[0])
        SEG = sitk.ReadImage(list(self.data.iloc[[item]]['SEG'])[0])
        Status = int(list(self.data.iloc[[item]]['Status'])[0])
        censorship = int(list(self.data.iloc[[item]]['censorship'])[0])
        survival_months = int(list(self.data.iloc[[item]]['survival_months'])[0])
        # ids = os.path.basename(os.path.dirname(list(self.data.iloc[[item]]['MRI'])[0]))
        label = int(list(self.data.iloc[[item]]['disc_label'])[0])
        

        ids = list(self.data.iloc[[item]]['Trial PatientID'])[0]
            
        image = sitk.GetArrayFromImage(image)
        SEG = sitk.GetArrayFromImage(SEG)
            
        sample = {'image': image, 'seg': SEG, 'Status':Status, 'censorship':censorship, 'survival_months':survival_months, 'ids':ids,'label':label }

        if self.mode == 'train':
            sample = transform(sample)
        if self.mode == 'valid':
            sample = transform_valid(sample)

        return sample
        
    def __len__(self):
        return len(self.data)