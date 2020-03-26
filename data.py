import numpy as np
import xml.etree.ElementTree as ET
import cv2
from PIL import Image

from utils import corners_to_center, center_to_corners

class DataLoader:

    def __init__(self, path, size_x=224, size_y=224):
        """
        description:
            init DataLoader, class to load segmentation data from Pascal VOC
        
        input:
            path: string, path to Pascal VOC dataset
            size_x: int, size to rescale width of images to
            size_y: int, size to rescale height of images to
        
        return:
            instance of DataLoader class
        """
        
        self.path = path
        self.size_x = size_x
        self.size_y = size_y
        
    def load_seg_lists(self):
        """
        description:
            loads list of ids of training and validation images for segmentation challenge
            
        input:
            NA
            
        return:
            tuple of the form (train_list, val_list)
                train_list: list of strings, each string is the id of a training image
                val_list: list of strings, each string is the id of a val image
        """
        
        train_path = self.path + 'ImageSets/Segmentation/train.txt'
        val_path = self.path + 'ImageSets/Segmentation/val.txt'
        
        with open(train_path, 'r') as f:
            train_str = f.read()
            
        with open(val_path, 'r') as f:
            val_str = f.read()
            
        train_list = train_str.split('\n')[:-1]  # last element is whitespace string
        val_list = val_str.split('\n')[:-1]  # last element is whitespace string

        return (train_list, val_list)
        
    def load_img(self, id, rescale=False):
        """
        description:
            receives as input an id and loads the image as numpy array
            
        input:
            id: string, id of image to load
            rescale: bool, default False. if true, rescales image so that height is 600
                width scales proportionally
                
        return:
            numpy array representing loaded image
        """
        
        img_path = self.path + 'JPEGImages/' + id + '.jpg'
        img = cv2.imread(img_path)
        if rescale:
            img = self.rescale_img(img)
        return img
        
    def rescale_img(self, img, nearest=False):
        """
        description:
            receives image and resizes according to height = self.size, width scales 
            proportionally
            
        input:
            img: np array of ints
            nearest: bool, default False, if True, use nearest neighbor interpolation
            
        return:
            np array of ints, representing resized image
        """
        
        width = self.size_x
        height = self.size_y
        if nearest:
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_NEAREST)
        else: 
            img = cv2.resize(img, (width, height))
        return img
        
    def load_seg(self, id, rescale=False, raw=False):
        """
        description:
            receives as input an id and loads the image segmentation as a numpy array
            
        input:
            id: string, id of image to load
            rescale: bool, default False. if True, rescales image
            raw: bool, default False, if True, loads colors shown in image rather than class as uint8
                
        return:
            numpy array representing loaded image (TODO: return shape)
        """

        img_path = self.path + 'SegmentationClass/' + id + '.png'
        if raw: 
            img = cv2.imread(img_path)
        else:
            img = np.array(Image.open(img_path))
        if rescale:
            img = self.rescale_img(img, nearest=True)
        return img
        
    def show_seg(self, id, rescale=False, time=5000):
        """
        description:
            receives as input an id and displays the image with segmented pixels shown
            
        input:
            id: string, id of image to show
            rescale: bool, default True, whether or not to rescale image to size used in model
            time: int, default 5000, time to display the image for in ms
            
        return:
            NA
        """
        
        img = self.load_img(id, rescale=True)
        target = self.load_seg(id, rescale=True, raw=True)
        
        blended = cv2.addWeighted(img, 0.2, target, 0.8, 0)
        
        cv2.imshow(str(id), blended)
        cv2.waitKey(time)
        cv2.destroyAllWindows()

def color_map(N=256, normalized=False):
    """
    description:
        calculate color map used to map Pascal VOC classes to colors used in images
        NOTE: I did not write this function myself
    
    input:
        N: int, number of colors
        normalize: bool, if True, return values between 0 and 1
    
    return:
        ndarray of ints of shape (N, 3) representing [b, g, r] values of each color
    """
    
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b, g, r])

    cmap = cmap/255 if normalized else cmap
    return cmap
