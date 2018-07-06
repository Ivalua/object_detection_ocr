import json
import glob
from lxml import etree
import numpy as np
import cv2
import math, os
import random
import gc
from . import compute_grids, compute_grids_
from keras.utils.data_utils import Sequence

class FlowGenerator(Sequence):
    def __init__(self, config, files, name, layer_offsets, layer_strides, layer_fields, resize, classes, target_size=(150, 150), batch_size=1, iou_treshold = .3, stride_margin= True): #rescale=1./255, shear_range=0.2, zoom_range=0.2, brightness=0.1, rotation=5.0, zoom=0.1

        self.config = config
        self.file_path_list = files
        self.name = name
        self.layer_offsets, self.layer_strides, self.layer_fields = layer_offsets, layer_strides, layer_fields
        self.classes = classes
        self.num_classes = len(classes)
        self.resize = resize
        self.img_h = target_size[0]
        self.img_w = target_size[1]
        self.batch_size = batch_size
        self.iou_treshold = iou_treshold
        self.stride_margin = stride_margin
        # self.brightness = brightness
        # self.rotation = rotation
        # self.zoom = zoom


    def __len__(self):
        return len(self.file_path_list) // self.batch_size

    def __getitem__(self, i):
        X = np.zeros((self.batch_size, self.img_h, self.img_w, 1), dtype='float32')
        groundtruth = [] # for mAP score computation and verification

        ns = {'d': self.config["namespace"]}
        for n, xml_file in enumerate(self.file_path_list[i*self.batch_size:(i+1)*self.batch_size]):
            # print(self.name, i, n, "/", self.batch_size, xml_file)
            root = etree.parse( xml_file )
            pages = root.findall(".//d:" + self.config["page_tag"], ns)
            # for p, page in enumerate(pages):
            p =0
            page = pages[0]
            page_size = (int(page.get("height")), int(page.get("width")))
            prefix = ""
            if len(pages) > 1:
                prefix = "-" + str(p)
            img_path = xml_file[:-4] + prefix + ".jpg"
            image = cv2.imread(img_path, 0)
            if (image is None) or (image.shape != page_size) :
                print("Read Error " + img_path)
                continue
            image = image / 255.

            f = 1.
            if self.resize != "":
                r1 = int(self.resize) / image.shape[0]
                r2 = int(self.resize) / image.shape[1]
                f = min(r1, r2)
                image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)

            # find a good random crop
            e = 0
            while e < 100:
                e = e +1
                if image.shape[1] > self.img_w:
                    x_ = np.random.choice(image.shape[1] - self.img_w)
                    w_ = self.img_w
                else:
                    x_ = 0
                    w_ = image.shape[1]
                if image.shape[0] > self.img_h:
                    y_ = np.random.choice(image.shape[0] - self.img_h)
                    h_ = self.img_h
                else:
                    y_ = 0
                    h_ = image.shape[0]
                chars = page.findall(".//d:" + self.config["char_tag"], ns)
                nb_chars = 0
                for c in chars:
                    x1 = float(c.get(self.config["x1_attribute"])) * f - x_
                    y1 = float(c.get(self.config["y1_attribute"])) * f - y_
                    x2 = float(c.get(self.config["x2_attribute"])) * f - x_
                    y2 = float(c.get(self.config["y2_attribute"])) * f - y_
                    if (x1 > 0) and (x2 < w_) and (y1 > 0) and (y2 < h_) :
                        nb_chars = nb_chars + 1
                if nb_chars > 10:
                    break

            X[n, :h_, :w_ , 0] = image[y_:y_+h_, x_:x_+w_]

            chars = page.findall(".//d:" + self.config["char_tag"], ns)
            # print("   Nb chars:", len(chars))
            for c in chars:
                x1 = int(float(c.get(self.config["x1_attribute"])) * f - x_)
                y1 = int(float(c.get(self.config["y1_attribute"])) * f - y_)
                x2 = int(float(c.get(self.config["x2_attribute"])) * f - x_)
                y2 = int(float(c.get(self.config["y2_attribute"])) * f - y_)
                if (x1 < 0) or (x2 > w_) or (y1 < 0) or (y2 > h_) or ( min(y2 - y1, x2 - x1)  <= 0.0  ): # or ( max(x2 - x1, y2 - y1) < (self.layer_fields[0] - self.layer_strides[0]) / 2 ):
                    continue
                # discard too small chars
                # if max(x2 - x1, y2 - y1) < 7:
                #     continue
                if c.text in self.classes:
                    groundtruth.append((i *self.batch_size + n, y1, x1, y2 - y1, x2 - x1, self.classes.index(c.text)))

        grids = compute_grids_(i *self.batch_size, self.batch_size, groundtruth, self.layer_offsets, self.layer_strides, self.layer_fields, (self.img_h, self.img_w), self.stride_margin, self.iou_treshold, self.num_classes)

        return X, grids #, np.array(groundtruth)

    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        random.shuffle(self.file_path_list)
        # Fix memory leak (Keras bug)
        gc.collect()



class Dataset:

    def __init__(self, name = "", batch_size=1, input_dim=1000, resize="", layer_offsets = [14], layer_strides = [28], layer_fields=[28], iou_treshold = .3, **kwargs):
        local_keys = locals()
        self.enable_classification = False
        self.enable_boundingbox = True
        self.enable_segmentation = False

        #classes
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", \
            "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", \
            "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", \
            "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "(", ")", "%"]
        self.num_classes = len(self.classes)
        print("Nb classes: " + str(self.num_classes))

        self.img_h = input_dim
        self.img_w = int(.7 * input_dim)
        self.input_shape = ( self.img_h, self.img_w , 1)
        self.stride_margin = True

        with open('datasets/document.conf') as config_file:
            config = json.load(config_file)
        xml_all_files = glob.glob(config["directory"] + "/*.xml")
        num_files = len(xml_all_files)
        num_train = int(0.9 * num_files)
        print("{} files in OCR dataset, split into TRAIN 90% and VAL 10%".format(num_files))
        xml_train_files = xml_all_files[0:num_train]
        xml_test_files = xml_all_files[num_train:]

        self.train = FlowGenerator(config, xml_train_files, "TRAIN", layer_offsets, layer_strides, layer_fields, resize, self.classes, target_size=(self.img_h, self.img_w),
                batch_size=batch_size, iou_treshold = iou_treshold, stride_margin= self.stride_margin) #, rescale=1./255, shear_range=0.2, zoom_range=0.2
        self.val = FlowGenerator(config, xml_test_files, "VAL", layer_offsets, layer_strides, layer_fields, resize, self.classes, target_size=(self.img_h, self.img_w),
                batch_size=batch_size, iou_treshold = iou_treshold, stride_margin= self.stride_margin) #, rescale=1./255
        self.test = FlowGenerator(config, xml_test_files, "TEST", layer_offsets, layer_strides, layer_fields, resize, self.classes, target_size=(self.img_h, self.img_w),
                batch_size=batch_size, iou_treshold = iou_treshold, stride_margin= self.stride_margin) #, rescale=1./255

        # for compatibility
        self.gt_test = []
        self.stride_margin = 0
