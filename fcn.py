import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import pandas as pd
import plotly.express as px
import random
from math import floor, ceil
import time
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import datetime

from data import DataLoader, color_map, PATH
from utils import iou_pixelwise, bilinear_kernel_init

class FCN:

    def __init__(self, path, size_x=448, size_y=448):
        """
        description:
            FCN class
        
        input:
            path: string, path to VOC PASCAL dataset
            size_x: int, width to scale images to
            size_y: int, height to scale images to
        
        return:
            NA
        """
        
        self.path = path
        self.size_x = size_x
        self.size_y = size_y
        self.n_classes = 21
        
        self.data = DataLoader(self.path, size_x=self.size_x, size_y=self.size_y)
        self.train_list, self.val_list = self.data.load_seg_lists()
        
        self.fcn = self.model_init()
        self.fcn.summary()
        
    def save(self, name):
        """
        description:
            save model weights
            
        input:
            name: string, prefix for model name
            
        return:
            NA
        """
        
        self.fcn.save_weights(name + '_fcn.h5')
        print('fcn saved with prefix {}'.format(name))
        
    def load(self, name):
        """
        description:
            load model weights
            
        input:
            name: string, prefix for model name
            
        return:
            NA
        """
        
        self.fcn.load_weights(name + '_fcn.h5')
        print('fcn with prefix {} loaded'.format(name))

    def model_init(self, lr=1e-5, alpha=0.0005):
        """
        description:
            initializes fcn model
        
        input:
            lr: float, learning rate
            alpha: float, weight regularization parameter
        
        return:
            keras.models.Model instance
        """
        
        vgg16 = keras.applications.vgg16.VGG16(include_top=False, 
                                               weights='imagenet', 
                                               input_tensor=None, 
                                               input_shape=(self.size_y, self.size_x, 3), 
                                               pooling=None, 
                                               classes=1000)
        
        # add regularization
        for i in [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17]:
            layer = vgg16.layers[i]
            layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda: keras.regularizers.l2(alpha)(layer.bias))
        
        input1 = vgg16.get_layer('input_1').input
        block5_pool_output = vgg16.get_layer('block5_pool').output
        
        vgg16_top = keras.applications.vgg16.VGG16(include_top=True, 
                                                   weights='imagenet', 
                                                   input_tensor=None, 
                                                   pooling=None, 
                                                   classes=1000)
        
        fc1 = vgg16_top.get_layer('fc1')
        fc2 = vgg16_top.get_layer('fc2')
        
        fc1_weights = fc1.get_weights()
        fc1_w = tf.reshape(fc1_weights[0], shape=(7, 7, 512, 4096))
        fc1_b = fc1_weights[1]
        conv6 = keras.layers.Conv2D(4096, 
                                    kernel_size = 7, 
                                    strides = 1, 
                                    padding = 'same',
                                    activation = 'relu',
                                    weights = [fc1_w, fc1_b],
                                    kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                    bias_regularizer = tf.keras.regularizers.l2(alpha),
                                    name = 'conv6')
        
        fc2_weights = fc2.get_weights()
        fc2_w = tf.reshape(fc2_weights[0], shape=(1, 1, 4096, 4096))
        fc2_b = fc2_weights[1]
        conv7 = keras.layers.Conv2D(4096, 
                                    kernel_size = 1, 
                                    strides = 1, 
                                    padding = 'same',
                                    activation = 'relu',
                                    weights = [fc2_w, fc2_b],
                                    kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                    bias_regularizer = tf.keras.regularizers.l2(alpha),
                                    name = 'conv7')
        
        conv7_to_cls = keras.layers.Conv2D(self.n_classes, 
                                           kernel_size = 1, 
                                           strides = 1, 
                                           padding = 'same',
                                           activation = 'relu',
                                           kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                           bias_regularizer = tf.keras.regularizers.l2(alpha),
                                           name = 'conv7_to_cls')
        
        pool4_to_cls = keras.layers.Conv2D(self.n_classes, 
                                           kernel_size = 1, 
                                           strides = 1, 
                                           padding = 'same',
                                           activation = 'relu',
                                           kernel_initializer = 'zeros',
                                           kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                           bias_regularizer = tf.keras.regularizers.l2(alpha),
                                           name = 'pool4_to_cls')
                                    
        pool3_to_cls = keras.layers.Conv2D(self.n_classes, 
                                           kernel_size = 1, 
                                           strides = 1, 
                                           padding = 'same',
                                           activation = 'relu',
                                           kernel_initializer = 'zeros',
                                           kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                           bias_regularizer = tf.keras.regularizers.l2(alpha),
                                           name = 'pool3_to_cls')
        
        conv_tr1 = keras.layers.Conv2DTranspose(self.n_classes, 
                                                3, 
                                                strides=2, 
                                                padding='same', 
                                                activation='relu', 
                                                weights = bilinear_kernel_init(3, self.n_classes),
                                                kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                                bias_regularizer = tf.keras.regularizers.l2(alpha),
                                                name='conv_tr1')
        
        conv_tr2 = keras.layers.Conv2DTranspose(self.n_classes, 
                                                3, 
                                                strides=2, 
                                                padding='same', 
                                                activation='relu', 
                                                weights = bilinear_kernel_init(3, self.n_classes),
                                                kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                                bias_regularizer = tf.keras.regularizers.l2(alpha),
                                                name='conv_tr2')
                                    
        conv_tr3 = keras.layers.Conv2DTranspose(self.n_classes, 
                                                3, 
                                                strides=2, 
                                                padding='same', 
                                                activation='relu', 
                                                weights = bilinear_kernel_init(3, self.n_classes),
                                                kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                                bias_regularizer = tf.keras.regularizers.l2(alpha),
                                                name='conv_tr3')
                                    
        conv_tr4 = keras.layers.Conv2DTranspose(self.n_classes, 
                                                3, 
                                                strides=2, 
                                                padding='same', 
                                                activation='relu', 
                                                weights = bilinear_kernel_init(3, self.n_classes),
                                                kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                                bias_regularizer = tf.keras.regularizers.l2(alpha),
                                                name='conv_tr4')
                                    
        conv_tr5 = keras.layers.Conv2DTranspose(self.n_classes, 
                                                3, 
                                                strides=2, 
                                                padding='same', 
                                                activation='relu', 
                                                weights = bilinear_kernel_init(3, self.n_classes),
                                                kernel_regularizer = tf.keras.regularizers.l2(alpha),
                                                bias_regularizer = tf.keras.regularizers.l2(alpha),
                                                name='conv_tr5')
        
        softmax = keras.layers.Softmax()
        
        block4_pool_output = vgg16.get_layer('block4_pool').output
        pool4_cls_output = pool4_to_cls(block4_pool_output)
        
        block3_pool_output = vgg16.get_layer('block3_pool').output
        pool3_cls_output = pool3_to_cls(block3_pool_output)
        
        upsample1_output = conv_tr1(conv7_to_cls(conv7(conv6(block5_pool_output))))
        upsample1_fused = keras.layers.Add()([upsample1_output, pool4_cls_output])
        
        upsample2_output = conv_tr2(upsample1_fused)
        upsample2_fused = keras.layers.Add()([upsample2_output, pool3_cls_output])
        
        output = softmax(conv_tr5(conv_tr4(conv_tr3(upsample2_fused))))

        fcn = keras.models.Model(inputs = input1, outputs = output)
        fcn.summary()

        optimizer = keras.optimizers.Adam(learning_rate=lr)

        fcn.compile(optimizer = optimizer, loss = self.fcn_loss)
        
        return fcn

    def fcn_loss(self, y_true, y_pred):
        """
        description:
            loss for fcn model
        
        input:
            y_true: tensor of shape (mb_size, height, width, n_classes) one-hot encoded
                note that 'ignore' pixels are coded as all zeros.
            y_pred: tensor of shape (mb_size, height, width, n_classes) with predicted probabilities
                of each class
        
        return:
            float, loss value
        """
        
        ignore = tf.reduce_sum(y_true, axis=-1)  # 0 if ignore, 1 if don't ignore
        categorical_crossentropy = keras.losses.CategoricalCrossentropy()
        return categorical_crossentropy(y_true, y_pred, sample_weight=ignore)
    
    def format_seg(self, seg):
        """
        description:
            one-hot encodes segmentation classes
        
        input:
            seg: ndarray of ints of shape (height, width) with values representing classes
        
        return:
            ndarray of shape (height, width, n_classes) representing one-hot encoding
        """
        
        oh = tf.one_hot(tf.cast(seg, dtype=tf.int64), self.n_classes)
        return oh

    def train(self, epochs=175, batch_size=20, validate=True, tensorboard=True):
        """
        description:
            trains fcn for specified number of epochs, and optionally writes to tensorboard
        
        input:
            epochs: int, number of epochs to train for
            batch_size: int, size of minibatch
            validate: bool, if True, calculates validation loss after each epoch
            tensorboard: bool, if True, sends loss to tensorboard after every epoch
        
        return:
            NA
        """
        
        if tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = './logs/' + current_time + '/train'
            val_log_dir = './logs/' + current_time + '/val'
            train_writer = tf.summary.create_file_writer(train_log_dir)
            val_writer = tf.summary.create_file_writer(val_log_dir)
        
        epoch_losses = []
        epoch_losses_val = []
        n_batches = len(self.train_list) // batch_size
        n_batches_validate = len(self.val_list) // batch_size
        for epoch in range(epochs):
            start_time = time.perf_counter()
            losses = []
            val_losses = []
            train_list = random.sample(self.train_list, len(self.train_list))
            
            for i in range(n_batches):
                X_list = []
                y_list = []
                for id in train_list[batch_size * i: batch_size * (i + 1)]:
                    img = self.data.load_img(id, rescale=True)
                    target = self.format_seg(self.data.load_seg(id, rescale=True))
                    X_list.append(img)
                    y_list.append(target)
                X = np.stack(X_list)
                y = np.stack(y_list)
                loss = self.fcn.train_on_batch(X, y)
                losses.append(loss)
                print('epoch: {}, batch: {}/{}, loss: {:0.6f}'.format(epoch + 1, i + 1, n_batches, np.mean(losses)), end='\r')
            
            epoch_time = int(time.perf_counter() - start_time)
            print('epoch: {}, batch: {}/{}, loss: {:0.6f}, epoch duration: {} seconds'.format(epoch + 1, i + 1, n_batches, np.mean(losses), epoch_time)) 
            epoch_losses.append(np.mean(losses))
            
            if tensorboard: 
                with train_writer.as_default():
                    tf.summary.scalar('loss', np.mean(losses), step = epoch + 1)
            
            if validate:
                start_time_val = time.perf_counter()
                
                for j in range(n_batches_validate):
                    X_list = []
                    y_list = []
                    for id in self.val_list[batch_size * j: batch_size * (j + 1)]:
                        img = self.data.load_img(id, rescale=True)
                        target = self.format_seg(self.data.load_seg(id, rescale=True))
                        X_list.append(img)
                        y_list.append(target)
                    X = np.stack(X_list)
                    y = np.stack(y_list)
                    loss = self.fcn.test_on_batch(X, y)
                    val_losses.append(loss)
                    print('epoch: {}, batch: {}/{}, val loss: {:0.6f}'.format(epoch + 1, j + 1, n_batches_validate, np.mean(val_losses)), end='\r')
                
                epoch_time_val = int(time.perf_counter() - start_time_val)
                print('epoch: {}, batch: {}/{}, val loss: {:0.6f}, val duration: {} seconds'.format(epoch + 1, j + 1, n_batches_validate, np.mean(val_losses), epoch_time_val))
                epoch_losses_val.append(np.mean(val_losses))
                
                if tensorboard: 
                    with val_writer.as_default():
                        tf.summary.scalar('loss', np.mean(val_losses), step = epoch + 1)
        
        loss_df = pd.DataFrame(data = zip(range(1, epochs + 1), epoch_losses), columns = ['epoch', 'loss'])
        fig = px.line(loss_df, x="epoch", y="loss", title='Training Loss')
        fig.show()
        
        if validate:
            loss_val_df = pd.DataFrame(data = zip(range(1, epochs + 1), epoch_losses_val), columns = ['epoch', 'loss'])
            fig = px.line(loss_val_df, x="epoch", y="loss", title='Validation Loss')
            fig.show()
        
    def predict(self, id, visualize=True, time=5000):
        """
        description:
            make prediction using image id, and optionally visualize
        
        input:
            id: string
            visualize: bool, if True, display prediction over image 
            time: int, length of time in ms to display image
        
        return:
            ndarray of shape (height, width) where values represent predicted class
        """
        
        img = self.data.load_img(id, rescale=True)
        X = np.expand_dims(img, axis=0)
        pred = self.fcn.predict(X)[0]
                
        if visualize: 
            pred_cls = np.argmax(pred, axis=-1)

            color_map_arr = color_map()
            pred_img = np.zeros(pred_cls.shape + (3, ), dtype='uint8')
            for i in range(pred_cls.shape[0]):
                for j in range(pred_cls.shape[1]):
                    pred_img[i, j, :] = color_map_arr[pred_cls[i, j]]
            
            blended = cv2.addWeighted(img, 0.2, pred_img, 0.8, 0)
            cv2.imshow(str(id), blended)
            cv2.waitKey(time)
            cv2.destroyAllWindows()
        
        return pred
    
    def evaluate(self, val=True):
        """
        description:
            calculates mean pixelwise IOU score over validation or training set
        
        input:
            val: bool, if True, uses validation set. otherwise, uses training set
        
        return:
            float, mean IOU score
        """
        
        evaluate_list = self.val_list if val else self.train_list
        tps, fps, fns = [], [], []
        i = 0
        for id in evaluate_list:
            i += 1
            pred = self.predict(id, visualize=False)
            pred_cls = np.argmax(pred, axis=-1)
            target = self.data.load_seg(id, rescale=True)
            tp, fp, fn = iou_pixelwise(pred_cls, target)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            print("calculating iou for image: {}/{}".format(i, len(evaluate_list)), end='\r')
        print("calculating iou for image: {}/{}".format(i, len(evaluate_list)))
        tp_arr = np.stack(tps)
        fp_arr = np.stack(fps)
        fn_arr = np.stack(fns)
        tp = np.sum(tp_arr, axis=0)
        fp = np.sum(fp_arr, axis=0)
        fn = np.sum(fn_arr, axis=0)
        iou_score = tp / (tp + fp + fn)
        for cls in range(self.n_classes):
            print("iou for class {}: {}".format(cls, iou_score[cls]))
        print("mean iou: {}".format(np.mean(iou_score)))
        return np.mean(iou_score)
