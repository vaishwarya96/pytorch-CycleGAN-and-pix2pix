import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, get_modified_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        # read a image given a random integer index
        AB_path = self.AB_paths[index]

        AB = cv2.imread(AB_path, -1)
        w, h, c = AB.shape
        #AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        #w, h = AB.size
        #w2 = int(w / 2)
        #A = AB.crop((0, 0, w2, h))
        #B = AB.crop((w2, 0, w, h))

        h2 = int(h/2)
        A = AB[0:w, 0:h2, :]
        B = AB[0:w, h2:h, :]
        
        #A = A
        #B = B
        
        mean = 21176.97653086236
        std_dev = 1876.2721426829492
        max_val = 65535
        min_val = 0

        #A = (A - mean)/std_dev
        #B = (B - mean)/std_dev

        A = A/255.0/255.0
        B = B/255.0/255.0
        #B = B/255.0/255.0
        A = 2*A - 1
        B = 2*B - 1
        A = A.astype(float)
        B = B.astype(float)
        #A = cv2.normalize(A, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #B = cv2.normalize(B, None, alpha = -1, beta = 1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print('shapes : ', A.shape, B.shape)
        #print('type :   ', type(A), type(B))
        #exit(0)
        #B.astype(np.uint16)

        #pixels = list(A.getdata())
        #print(pixels)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.shape)
        
        if self.opt.phase == 'train':
            self.opt.color_jitter = False
            A_transform = get_modified_transform(self.opt, A, transform_params, grayscale=(self.input_nc == 1))
            self.opt.color_jitter = False
            B_transform = get_modified_transform(self.opt, B, transform_params, grayscale=(self.output_nc == 1))

        else:
            self.opt.color_jitter = False
            A_transform = get_modified_transform(self.opt, A, transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_modified_transform(self.opt, B, transform_params, grayscale=(self.output_nc == 1))
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

       # norm_1 = transforms.Normalize((0.5,), (0.5,))
       # norm_3 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        norm_1 = transforms.Normalize((mean,),(std_dev,))
        norm_3 = transforms.Normalize((mean,mean,mean),(std_dev,std_dev,std_dev))
        A_transform.astype(float)
        B_transform.astype(float)
        if self.opt.input_nc == 1:
            A = torch.from_numpy(A_transform).float().unsqueeze(0)
            #A = norm_1(A)
        else:
            A = torch.from_numpy(A_transform.transpose((2,0,1))).float()
            #A = norm_3(A)

        if self.opt.output_nc == 1:
            B = torch.from_numpy(B_transform).float().unsqueeze(0)
            #B = norm_1(B)
        else:
            B = torch.from_numpy(B_transform.transpose((2,0,1))).float()
            #B = norm_3(B)
        
       # A = A_transform(A)
       # B = B_transform(B)
        '''
        print("min A", torch.min(A))
        print("min B", torch.min(B))
        print("max A", torch.max(A))
        print("max B", torch.max(B))
        '''
        #print(A)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
