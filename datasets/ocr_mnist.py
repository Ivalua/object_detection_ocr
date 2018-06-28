from __future__ import division
import keras
from keras.datasets import mnist
import numpy as np
import os
from skimage import transform
from . import compute_grids, compute_grids_, compute_grids_local, load_from_local_file, save_to_local_file
import math

class Dataset:

    def __init__(self, name,  layer_offsets = [14, 28], layer_strides = [28, 56], layer_fields=[28, 56],
        input_dim=700, resize="", white_prob = 0., bb_positive="iou-treshold" , iou_treshold = .3, save=True, noise=False, **kwargs):
        local_keys = locals()
        self.enable_classification = False
        self.enable_boundingbox = True
        self.enable_segmentation = False

        if resize == "":
            digit_dim = [["28"]]
        else:
            digit_dim = [r.split("-") for r in resize.split(",")]

        assert len(layer_offsets) == len(digit_dim), "Number of layers in network do not match number of digit scales"

        self.img_h = int(input_dim)
        self.img_w = int(input_dim * .6)
        self.input_shape = ( self.img_h, self.img_w , 1)

        grid_dim = int(digit_dim[0][-1])
        nb_images_y = self.img_h  // grid_dim
        nb_images_x = self.img_w  // grid_dim

        self.num_classes = 10
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.stride_margin= False

        if load_from_local_file(**local_keys):
            return

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        if noise:
            NUM_DISTORTIONS_DB = 100000
            num_distortions=80
            distortions = []
            dist_size = (9, 9)
            all_digits = x_train.reshape([-1, 28, 28])
            num_digits = all_digits.shape[0]
            for i in range(NUM_DISTORTIONS_DB):
                rand_digit = np.random.randint(num_digits)
                rand_x = np.random.randint(28-dist_size[1])
                rand_y = np.random.randint(28-dist_size[0])

                digit = all_digits[rand_digit]
                distortion = digit[rand_y:rand_y + dist_size[0],
                                   rand_x:rand_x + dist_size[1]]
                assert distortion.shape == dist_size
                #plt.imshow(distortion, cmap='gray')
                #plt.show()
                distortions += [distortion]
            print("Created distortions")

            def add_distortions(image):
                canvas = np.zeros_like(image)
                for i in range(num_distortions):
                    rand_distortion = distortions[np.random.randint(NUM_DISTORTIONS_DB)]
                    rand_x = np.random.randint(image.shape[1]-dist_size[1])
                    rand_y = np.random.randint(image.shape[0]-dist_size[0])
                    canvas[rand_y:rand_y+dist_size[0],
                           rand_x:rand_x+dist_size[1], 0] = - rand_distortion
                canvas += image
                return np.clip(canvas, 0, 1)


        def create_tile(x, y):
            total_digits = x.shape[0]
            nb_images = int(total_digits / nb_images_x / nb_images_y / (1-white_prob) )
            tiles = np.ones( [nb_images, self.img_h , self.img_w, x.shape[3]], dtype = 'float32')
            occupations = np.zeros( [nb_images, self.img_h , self.img_w, 1], dtype = 'float32')
            groundtruth = [] # for mAP score computation and verification

            i = 0
            for tile in range(nb_images):
                for s in reversed(range(len(layer_offsets))):
                    nb_samples = 0
                    img_dim = digit_dim[s]
                    anchor_dim = layer_fields[s]
                    while nb_samples < (1. - white_prob) * nb_images_x * nb_images_y / len(digit_dim) * grid_dim / int(img_dim[-1]):
                        # pick a random row, col on the scale grid
                        row = np.random.choice(nb_images_y )
                        col = np.random.choice(nb_images_x )
                        if len(img_dim) > 1:
                            dim = int(math.ceil(int(img_dim[0]) + np.random.rand() * (int(img_dim[1]) - int(img_dim[0])) ))
                        else:
                            dim = int(img_dim[0])
                        xc = (col + .5) * grid_dim
                        yc = (row + .5) * grid_dim
                        x_ = int(xc - dim/2)
                        y_ = int(yc - dim/2)
                        x_range = slice(x_, x_ + dim)
                        y_range = slice(y_, y_ + dim)
                        if (x_ < 0) or (y_ < 0) or (x_ + dim > self.img_w) or (y_ + dim > self.img_h):
                            continue

                        # if position available add it
                        if np.sum(occupations[ tile, y_range, x_range, 0 ]) == 0.:
                            resized_x = transform.resize(x[i], (dim, dim), mode='constant')
                            tiles[ tile, y_range, x_range, ...] = 1.0 - resized_x # change for white background
                            groundtruth.append((tile, y_, x_, dim, dim, np.argmax(y[i])))
                            occupations[ tile, y_range, x_range, ...] = 1.0
                            i = (i + 1) % total_digits
                            nb_samples = nb_samples + 1

                if noise:
                    tiles[ tile ] = add_distortions(tiles[ tile ])

            import time
            now = time.time()
            grids = compute_grids(0, nb_images, groundtruth, layer_offsets, layer_strides, layer_fields, self.input_shape, self.stride_margin, iou_treshold, self.num_classes, bb_positive="iou-treshold")
            if False: # timing eval
                t1 = time.time() -now
                now = time.time()
                grids2 = compute_grids_local(0, nb_images, groundtruth, layer_offsets, layer_strides, layer_fields, self.input_shape, self.stride_margin, iou_treshold, self.num_classes)
                print("grids", t1)
                print("grids2", time.time() -now)
                print("is nan", np.isnan(grids2[0].min()))
                for s in range(len(grids)):
                    for l in range(grids[s].shape[-1]):
                        print(np.allclose(grids[s][...,l],grids2[s][...,l]))
                print(np.allclose(np.argmax(grids[0][...,:10], axis=3), np.argmax(grids2[0][...,:10], axis=3)))

            return tiles, grids, np.array(groundtruth)

        x_train, y_train, gt_train = create_tile(x_train, y_train)
        x_test, y_test, gt_test = create_tile(x_test, y_test)

        self.x_train = x_train
        self.y_train = y_train
        self.gt_train = gt_train
        self.x_test = x_test
        self.y_test = y_test
        self.gt_test = gt_test

        save_to_local_file(**local_keys)

        # from skimage.io import imsave
        # import os
        # if not os.path.exists("logs"): #args.logs
        #     os.mkdir("logs")
        # image_id = 1
        # image = (x_train[image_id, :, :, 0] * 255.).astype(np.uint8)
        # imsave(os.path.join("logs", str(image_id) + '_input.png'), image)
