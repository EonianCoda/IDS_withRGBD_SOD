import cv2
import torchvision.transforms as transforms
import numpy as np
from fast_slic import Slic
import config
from PIL import Image

class SalObjDataset_test(object):
    def __init__(self, testsize=352, output_size=(640, 480)):
        self.testsize = testsize
        self.output_size = output_size
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)), 
                                                    transforms.ToTensor()])
        

    def get(self, img_path:str, depth_path:str):
        image = self.rgb_loader(img_path)
       
        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR)
        
        slic = Slic(num_components=config.TRAIN['num_superpixels'], compactness=10)
        SS_map = slic.iterate(np_img)
        
        SS_map = SS_map + 1

        SS_maps = []

        for i in range(1, config.TRAIN['num_superpixels'] + 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1
            SS_maps.append(buffer)
        
        ss_map = np.array(SS_maps)
        image = self.transform(image)
        depth = self.binary_loader(depth_path)

        depth = Image.fromarray(255 - np.array(depth))

        np_depth = np.array(depth)
        np_depth = cv2.resize(np_depth, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR)
        np_depth = cv2.cvtColor(np_depth, cv2.COLOR_GRAY2BGR)

        depth_SS_map = slic.iterate(np_depth)

        depth_SS_map = depth_SS_map + 1

        depth_SS_maps = []

        for i in range(1, config.TRAIN['num_superpixels'] + 1):
            buffer = np.copy(depth_SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1
            depth_SS_maps.append(buffer)
        
        depth_ss_map = np.array(depth_SS_maps)
        
        depth = self.depths_transform(depth)
        
        return image, depth, self.output_size, ss_map, depth_ss_map


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')