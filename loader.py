import os
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms


class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def folder_content_getter(datadir, data, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')
    if data == 'sketchy':
        shot_dir = "zeroshot0"
        if skim == 'sketch':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/all_photo_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')
    elif data == 'sketchy2':
        shot_dir = "zeroshot1"
        if skim == 'sketch':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/all_photo_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif data == 'tuberlin':
        if skim == 'sketch':
            file_ls_file = datadir + '/TUBerlin/zeroshot/png_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = datadir + '/TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif data == 'quickdraw':
        if skim == 'sketch':
            file_ls_file = datadir + '/QuickDraw/zeroshot/sketch_train.txt'
        elif skim == 'image':
            file_ls_file = datadir + '/QuickDraw/zeroshot/all_photo_train.txt'
        else:
            NameError(skim + ' not implemented!')

    else:
        NameError(data + ' not implemented! ')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()
    image_path_list = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    image_cate_list = np.array([int(ff.strip().split()[-1]) for ff in file_content])

    return image_path_list, image_cate_list

def zeroshot_getter(datadir, data, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')

    if data == 'sketchy':
        shot_dir = "zeroshot0"
        if skim == 'sketch':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_zero.txt'
        elif skim == 'image':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/all_photo_filelist_zero.txt'
        else:
            NameError(skim + ' not implemented!')
    elif data == 'sketchy2':
        shot_dir = "zeroshot1"
        if skim == 'sketch':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_zero.txt'
        elif skim == 'image':
            file_ls_file = datadir + f'/Sketchy/{shot_dir}/all_photo_filelist_zero.txt'
        else:
            NameError(skim + ' not implemented!')

    elif data == 'tuberlin':
        if skim == 'sketch':
            file_ls_file = datadir + '/TUBerlin/zeroshot/png_ready_filelist_zero.txt'
        elif skim == 'image':
            file_ls_file = datadir + '/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt'
        else:
            NameError(skim + ' not implemented!')

    elif data == 'quickdraw':
        if skim == 'sketch':
            file_ls_file = datadir + '/QuickDraw/zeroshot/sketch_zero.txt'
        elif skim == 'image':
            file_ls_file = datadir + '/QuickDraw/zeroshot/all_photo_zero.txt'
        else:
            NameError(skim + ' not implemented!')

    else:
        NameError(data + ' not implemented! ')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()
    image_path_list = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    image_cate_list = np.array([int(ff.strip().split()[-1]) for ff in file_content])

    return image_path_list, image_cate_list

class TestDataset(data.Dataset):
    def __init__(self,
                 data):

        datadir = './dataset'
        if data == 'sketchy' or data == 'sketchy2':
            self.instancedir = os.path.join(datadir, 'Sketchy')
        elif data == 'tuberlin':
            self.instancedir = os.path.join(datadir, 'TUBerlin')
        elif data == 'quickdraw':
            self.instancedir = os.path.join(datadir, 'QuickDraw')
        else:
            print('PathError')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                         transforms.Resize(224),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ])

        self.image_paths_A, self.image_cates_A = zeroshot_getter(datadir, data, skim='sketch')
        self.image_paths_B, self.image_cates_B = zeroshot_getter(datadir, data, skim='image')
        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):

        index_A = np.mod(index, self.domainA_size)
        index_B = np.mod(index, self.domainB_size)

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        image_A = self.transform(Image.open(os.path.join(self.instancedir,image_path_A)).convert('RGB'))
        image_B = self.transform(Image.open(os.path.join(self.instancedir,image_path_B)).convert('RGB'))

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]

        return image_A, index_A, target_A, image_B, index_B, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)

class EvalDataset(data.Dataset):
    def __init__(self,
                 data):

        datadir = './dataset'
        if data == 'sketchy' or data == 'sketchy2':
            self.instancedir = os.path.join(datadir, 'Sketchy')
        elif data == 'tuberlin':
            self.instancedir = os.path.join(datadir, 'TUBerlin')
        elif data == 'quickdraw':
            self.instancedir = os.path.join(datadir, 'QuickDraw')
        else:
            print('PathError')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ])

        self.image_paths_A, self.image_cates_A = folder_content_getter(datadir, data, skim='sketch')
        self.image_paths_B, self.image_cates_B = folder_content_getter(datadir, data, skim='image')
        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):

        index_A = np.mod(index, self.domainA_size)
        index_B = np.mod(index, self.domainB_size)

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        image_A = self.transform(Image.open(os.path.join(self.instancedir,image_path_A)).convert('RGB'))
        image_B = self.transform(Image.open(os.path.join(self.instancedir,image_path_B)).convert('RGB'))

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]

        return image_A, index_A, target_A, image_B, index_B, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)


class TrainDataset(data.Dataset):
    def __init__(self,
                 data,
                 aug_plus, dino_transform=None, sketch_transform=None):
        datadir='./dataset'
        if data == 'sketchy' or data == 'sketchy2':
            self.instancedir=os.path.join(datadir,'Sketchy')
        elif data == 'tuberlin':
            self.instancedir=os.path.join(datadir,'TUBerlin')
        elif data == 'quickdraw':
            self.instancedir=os.path.join(datadir,'QuickDraw')
        else:
            print('PathError')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if aug_plus:
            self.transform = dino_transform
            self.sketch_transform = sketch_transform
            print('Use dino aug ! ! !')
            # self.transform = transforms.Compose(
            #     [
            #         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            #         transforms.RandomApply([
            #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #         ], p=0.8),
            #         transforms.RandomGrayscale(p=0.2),
            #         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         normalize
            #     ]
            # )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )

        self.image_paths_A, self.image_cates_A = folder_content_getter(datadir, data, skim='sketch')
        self.image_paths_B, self.image_cates_B = folder_content_getter(datadir, data, skim='image')

        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):

        if index >= self.domainA_size:
            index_A = random.randint(0, self.domainA_size - 1)
        else:
            index_A = index

        if index >= self.domainB_size:
            index_B = random.randint(0, self.domainB_size - 1)
        else:
            index_B = index

        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]

        x_A = Image.open(os.path.join(self.instancedir,image_path_A)).convert('RGB')
        q_A = self.sketch_transform(x_A)
        # k_A = self.transform(x_A)

        x_B = Image.open(os.path.join(self.instancedir,image_path_B)).convert('RGB')
        q_B = self.transform(x_B)
        # k_B = self.transform(x_B)

        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]

        return q_A, index_A, q_B, index_B, target_A, target_B

    def __len__(self):

        return max(self.domainA_size, self.domainB_size)
