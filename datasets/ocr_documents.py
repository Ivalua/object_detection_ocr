import json
import glob
from lxml import etree
import numpy as np
# from scipy import misc
import cv2
from . import load_from_local_file, save_to_local_file, compute_grids, compute_grids_
import math, os

class Dataset:

    def __init__(self, name = "", input_dim=700, resize="", layer_offsets = [14], layer_strides = [28], layer_fields=[28], iou_treshold = .3, save=True, **kwargs):
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
        self.img_w = int(.6 * input_dim)
        self.input_shape = ( self.img_h, self.img_w , 1)
        self.stride_margin = True

        if load_from_local_file(**local_keys):
            return

        with open('datasets/document.conf') as config_file:
            config = json.load(config_file)

        xml_all_files = glob.glob(config["directory"] + "/*.xml")
        # xml_all_files = xml_all_files[:100]
        num_files = len(xml_all_files)
        num_train = int(0.9 * num_files)
        print("{} files in OCR dataset, split into TRAIN 90% and VAL 10%".format(num_files))
        xml_train_files = xml_all_files[0:num_train]
        xml_test_files = xml_all_files[num_train:]

        def create_dataset(xml_files, name):
            nb_images = len(xml_files)
            print("{} files in {} dataset".format(nb_images, name))
            groundtruth = []
            tiles = np.ones( [nb_images, self.img_h , self.img_w, 1], dtype = 'float32')
            ns = {'d': config["namespace"]}
            i = 0
            for xml_file in xml_files:
                if i >= nb_images:
                    break
                root = etree.parse( xml_file )
                print("{}/{} - {}".format(i, nb_images, xml_file))

                pages = root.findall(".//d:" + config["page_tag"], ns)
                for p, page in enumerate(pages):
                    if i >= nb_images:
                        break

                    prefix = ""
                    if len(pages) > 1:
                        prefix = "-" + str(p)
                    img_path = xml_file[:-4] + prefix + ".jpg"
                    image = cv2.imread(img_path, 0)

                    if (image is None) or (image.shape != (int(page.get("height")), int(page.get("width")))) :
                        print("Read Error " + img_path)
                        continue

                    image = image / 255.

                    f = 1.
                    if resize != "":
                        r1 = int(resize) / image.shape[0]
                        r2 = int(resize) / image.shape[1]
                        f = min(r1, r2)
                        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
                    print(image.shape)

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
                        chars = page.findall(".//d:" + config["char_tag"], ns)
                        nb_chars = 0
                        for c in chars:
                            x1 = float(c.get(config["x1_attribute"])) * f - x_
                            y1 = float(c.get(config["y1_attribute"])) * f - y_
                            x2 = float(c.get(config["x2_attribute"])) * f - x_
                            y2 = float(c.get(config["y2_attribute"])) * f - y_
                            if (x1 > 0) and (x2 < w_) and (y1 > 0) and (y2 < h_) :
                                nb_chars = nb_chars + 1
                        if nb_chars > 10:
                            break

                    tiles[i, :h_, :w_, 0] = image[y_:y_+h_, x_:x_+w_]

                    chars = page.findall(".//d:" + config["char_tag"], ns)
                    for c in chars:
                        x1 = float(c.get(config["x1_attribute"])) * f - x_
                        y1 = float(c.get(config["y1_attribute"])) * f - y_
                        x2 = float(c.get(config["x2_attribute"])) * f - x_
                        y2 = float(c.get(config["y2_attribute"])) * f - y_
                        if (x1 < 0) or (x2 > w_) or (y1 < 0) or (y2 > h_) or ( min(y2 - y1, x2 - x1)  <= 0.0 ): # or ( max(x2 - x1, y2 - y1) < (layer_fields[0] - layer_strides[0]) / 2 )
                            continue
                        # discard too small chars
                        # if max(x2 - x1, y2 - y1) < 7:
                        #     continue
                        if (c.text in self.classes):
                            groundtruth.append((i, y1, x1, y2 - y1, x2 - x1, self.classes.index(c.text)))

                    i = i + 1

            grids = compute_grids_(0, nb_images, groundtruth, layer_offsets, layer_strides, layer_fields, self.input_shape, self.stride_margin, iou_treshold, self.num_classes)
            return tiles, grids, np.array(groundtruth)

        x_train, y_train, gt_train = create_dataset(xml_train_files, "TRAIN")
        x_test, y_test, gt_test = create_dataset(xml_test_files, "TEST")

        self.x_train = x_train
        self.y_train = y_train
        self.gt_train = gt_train
        self.x_test = x_test
        self.y_test = y_test
        self.gt_test = gt_test

        save_to_local_file(**local_keys)
