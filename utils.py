import numpy as np
import tensorflow as tf

def corners_to_center(arr):
    """
    description:
        converts from corner coordinates to center coordinates
        
    input:
        box: ndarray of ints of shape (..., 4) with last axis of form [xmin xmax ymin ymax]
        
    return:
        ndarray of ints of same shape as box, with last axis of the form [x, y, w, h]
            x: x center
            y: y center
            w: width
            h: height
    """
    xmin, xmax, ymin, ymax = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + w/2
    y = ymin + h/2

    return np.stack([x, y, w, h], axis=-1).astype(np.int)
    
def center_to_corners(arr, yxyx=False, as_tf=False):
    """
    description:
        converts from center coordinates to corner coordinates
        
    input:
        arr: ndarray of ints of shape (..., 4) with last axis of the form [x y w h]
            x: x center
            y: y center
            w: width
            h: height
        
    return:
        ndarray of ints of the same shape as arr with last axis of the 
            form [xmin, xmax, ymin, ymax]
    """

    if as_tf:
        return center_to_corners_tf(arr, yxyx=yxyx)
    
    assert arr.shape[-1] == 4
    assert (arr[..., -1] >= 0).all()
    assert (arr[..., -2] >= 0).all()
    
    x, y, w, h = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    
    return np.stack([xmin, xmax, ymin, ymax], axis=-1).astype(np.int)
    
def center_to_corners_tf(tensor, yxyx=False):
    """
    description:
        converts from center coordinates to corner coordinates
        
    input:
        tensor: tensor of shape (..., 4) with last axis of the form [x y w h]
            x: x center
            y: y center
            w: width
            h: height
        yxyx: bool, default False, if True, formats output [ymin, xmin, ymax, xmax]
            otherwise formats output [xmin, xmax, ymin, ymax]
        
    return:
        tensor of the same shape as tensor with last axis of the form [ymin xmin ymax xmax]
    """

    assert tensor.shape[-1] == 4
    
    x, y, w, h = tensor[..., 0], tensor[..., 1], tensor[..., 2], tensor[..., 3]
    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    
    if yxyx:
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        
    else:
        return tf.stack([xmin, xmax, ymin, ymax], axis=-1)

def iou_pixelwise(pred, target, n_classes=21, ignore=255):
    """
    description:
        calculates pixelwise iou of each class between prediction and target
    
    input:
        pred: ndarray of uint8 of shape (height, width) where classes range from 0 to n_classes - 1
        target: ndarray of uint8 of shape (height, width) where classes range from 0 to n_classes - 1
        n_classes: int, default 21
        ignore: int, default 255, represents cls value to ignore
    
    return:
        tuple (tp, fp, fn):
            tp: ndarray of ints of shape (n_classes, ) representing number of true positives for each class
            fp: ndarray of ints of shape (n_classes, ) representing number of false positives for each class
            fn: ndarray of ints of shape (n_classes, ) representing number of false negatives for each class
    """
    
    tps, fps, fns = [], [], []
    for cls in range(n_classes):
        tp = (pred == cls) * (target == cls)
        fp = (pred == cls) * (target != cls) * (target != ignore)
        fn = (pred != cls) * (target == cls)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    tp = np.stack(tps, axis=-1)
    fp = np.stack(fps, axis=-1)
    fn = np.stack(fns, axis=-1)
    tp = np.sum(np.sum(tp, axis=0), axis=0)
    fp = np.sum(np.sum(fp, axis=0), axis=0)
    fn = np.sum(np.sum(fn, axis=0), axis=0)
    return (tp, fp, fn)

def bilinear_kernel_init(filter_size, num_channels):
    """
    description:
        generates weights for conv transpose layter to initialize with bilinear upsampling
        NOTE: I did not write this function myself
    
    inputs:
        filter_size: int, length of square filter
        num_channels: int, number of channels
    
    return:
        list of ndarrays [weights, biases]
    """
    
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
                                   
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel
    return [weights, np.zeros((num_channels, ))]
