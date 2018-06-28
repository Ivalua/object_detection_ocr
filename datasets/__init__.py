import os
import math
import numpy as np


## Utilities to save or load processed data

def get_outfile_name(a):
    if not os.path.exists("/sharedfiles/datasets"):
        os.makedirs("/sharedfiles/datasets")
    outfile = "/sharedfiles/datasets/" + a.get("name")

    noise = a.get("noise")
    if noise:
        outfile = outfile + "_noise"

    input_dim = a.get("input_dim")
    resize = a.get("resize")
    outfile = outfile + "_r" + str(resize) + "_i" + str(input_dim)

    layer_offsets = a.get("layer_offsets")
    layer_strides = a.get("layer_strides")
    layer_fields = a.get("layer_fields")
    for s in range(len(layer_offsets)):
        outfile = outfile + "_o" + str(layer_offsets[s]) + "_s" + str(layer_strides[s]) + "_f" + str(layer_fields[s])
    return outfile + "_iou" + str(a.get("iou_treshold")) + ".npz"


def load_from_local_file(**kwargs):
    outfile = get_outfile_name(kwargs)

    if os.path.exists(outfile):
        print("Loading from file " + outfile)
        layer_offsets = kwargs.get("layer_offsets")
        data = np.load(outfile)
        kwargs.get("self").x_train = data['arr_0']
        kwargs.get("self").gt_train = data['arr_1']
        kwargs.get("self").x_test = data['arr_2']
        kwargs.get("self").gt_test = data['arr_3']
        kwargs.get("self").y_train = []
        kwargs.get("self").y_test = []
        for i in range(len(layer_offsets)):
            kwargs.get("self").y_train.append(data[ "arr_" + str( 4 + i ) ])
            kwargs.get("self").y_test.append(data[ "arr_" + str( 4 + len(layer_offsets) + i )])
        return True
    return False

def save_to_local_file(**kwargs):
    if kwargs.get("save"):
        outfile = get_outfile_name(kwargs)
        print("Saving to file " + outfile)
        data = [kwargs.get("self").x_train, kwargs.get("self").gt_train, kwargs.get("self").x_test, kwargs.get("self").gt_test ] + kwargs.get("self").y_train + kwargs.get("self").y_test
        np.savez(outfile, *data)


## Utilities to compute outputs

def get_layer_sizes(img_shape=(10,10), layer_offsets=[2], layer_strides=[2]):
    sizes = []
    for l in range(len(layer_offsets)):
        nb_y = int(math.floor( (img_shape[0] -  layer_offsets[l] * 2 ) / layer_strides[l] )) + 1
        nb_x = int(math.floor( (img_shape[1] -  layer_offsets[l] * 2 ) / layer_strides[l] )) + 1
        sizes.append([nb_y, nb_x])
    return sizes

def get_position_on_grid(pos, layer_offset=2, layer_stride=2, layer_size=(10,10)):
    if pos[1] < layer_offset:
        x = 0
    else:
        x = round( (pos[1]-layer_offset)/layer_stride )

    if pos[0] < layer_offset:
        y = 0
    else:
        y = round( (pos[0]-layer_offset)/layer_stride )

    return min(y, layer_size[0] - 1), min(x, layer_size[1] - 1)


def get_positioned_anchors(img_shape=(10,10), layer_offsets=[2], layer_strides=[2], layer_fields=[2], margin = False):
    result = []
    for l in range(len(layer_offsets)):
        dim = layer_fields[l]
        if margin:
            dim = dim - layer_strides[l]

        nb_y = int(math.floor( (img_shape[0] -  layer_offsets[l] * 2 ) / layer_strides[l] )) + 1
        nb_x = int(math.floor( (img_shape[1] -  layer_offsets[l] * 2 ) / layer_strides[l] )) + 1
        positioned_anchors = np.zeros((nb_y, nb_x, 4))
        for i in range(nb_y):
            for j in range(nb_x):
                positioned_anchors[i, j, 0] = layer_offsets[l] + layer_strides[l] * i - dim / 2 # y
                positioned_anchors[i, j, 1] = layer_offsets[l] + layer_strides[l] * j - dim / 2 #x

                positioned_anchors[i, j, 2] = dim # h
                positioned_anchors[i, j, 3] = dim # w
        result.append(positioned_anchors)
    return result

def iou(a, b):
    intersection = np.maximum((np.minimum(a[..., 0] + a[...,2], b[...,0] + b[...,2]) - np.maximum(a[...,0], b[...,0])), 0) * \
        np.maximum((np.minimum(a[...,1] + a[...,3], b[...,1] + b[...,3]) - np.maximum(a[...,1], b[...,1])), 0)
    union = (np.maximum(a[...,0] + a[...,2], b[...,0] + b[...,2]) - np.minimum(a[...,0], b[...,0])) * \
        (np.maximum(a[...,1] + a[...,3], b[...,1] + b[...,3]) - np.minimum(a[...,1], b[...,1]) )
    return intersection / union


def compute_grids(start_index, nb_images, groundtruth, layer_offsets, layer_strides, layer_fields, input_shape, stride_margin, iou_treshold, num_classes, bb_positive="iou-treshold"):
    print("Computing grids...")
    anchors = get_positioned_anchors(input_shape, layer_offsets, layer_strides, layer_fields, stride_margin)
    layer_sizes = get_layer_sizes(input_shape, layer_offsets, layer_strides)

    grids = [] # create different grids for different downsizing factors
    # num_classes, iou > thresh, x, y, w, h, distance
    for layer_size in layer_sizes:
        grid = np.zeros( [nb_images] + layer_size + [num_classes+6], dtype = 'float32')
        grid[..., num_classes+5] = 100000 # initiate distances to high values
        grids.append(grid)
    collisions = 0
    pos = np.zeros([len(layer_sizes)]) # statistics

    for box in groundtruth:
        i = box[0] - start_index
        cl = box[5]
        for s in range(len(layer_sizes)):
            # we consider only letters smaller than this dimension
            # and a dimension smaller than the layer field
            # to ensure that 1) at least one anchor sees the full letter
            # 2) the layer field captures the blank around the letter
            # See margin=True in get_positioned_anchors method
            dim = layer_fields[s]
            if stride_margin:
                dim = dim - layer_strides[s]

            if (box[4] <= dim) and (box[3] <= dim) :
                if bb_positive == "in-anchor":
                    layer_y = max( min( ( box[1] + box[3]/2 - layer_offsets[s]) // layer_strides[s], layer_sizes[s][0] -1), 0)
                    layer_x = max( min( ( box[2] + box[4]/2 - layer_offsets[s]) // layer_strides[s], layer_sizes[s][1] -1), 0)
                    grids[s][ i, layer_y, layer_x, :num_classes ] = 0.0
                    grids[s][ i, layer_y, layer_x, cl ] = 1.0
                    grids[s][ i, layer_y, layer_x, num_classes ] = 1.0
                    grids[s][ i, layer_y, layer_x, num_classes + 1 ] = (box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2) / dim # y
                    grids[s][ i, layer_y, layer_x, num_classes + 2 ] = (box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) / dim # x
                    grids[s][ i, layer_y, layer_x, num_classes + 3 ] = math.log(dim/box[3], 2)
                    grids[s][ i, layer_y, layer_x, num_classes + 4 ] = math.log(dim/box[4], 2)
                    pos[s] += 1
                    break
                elif bb_positive == "iou-treshold":
                    scores = iou(anchors[s], np.tile(np.array(box[1:5]), (anchors[s].shape[0], anchors[s].shape[1], 1) ))
                    for layer_y in range(scores.shape[0]):
                        for layer_x in range(scores.shape[1]):
                            d = ( box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2)**2 + ( box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) **2
                            if scores[layer_y, layer_x] > iou_treshold:
                                if grids[s][ i, layer_y, layer_x, num_classes ] == 1.:
                                    collisions = collisions + 1
                                if d < grids[s][i, layer_y, layer_x, num_classes + 5] :
                                    pos[s] += 1
                                    grids[s][ i, layer_y, layer_x, :num_classes ] = 0.0
                                    grids[s][ i, layer_y, layer_x, cl ] = 1.0
                                    grids[s][ i, layer_y, layer_x, num_classes ] = 1.0
                                    grids[s][ i, layer_y, layer_x, num_classes + 1 ] = ( box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2) / dim # y
                                    grids[s][ i, layer_y, layer_x, num_classes + 2 ] = ( box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) / dim # x
                                    grids[s][ i, layer_y, layer_x, num_classes + 3 ] = math.log(dim/box[3], 2)
                                    grids[s][ i, layer_y, layer_x, num_classes + 4 ] = math.log(dim/box[4], 2)
                            if d < grids[s][i, layer_y, layer_x, num_classes + 5]:
                                grids[s][i, layer_y, layer_x, num_classes + 5] = d
                    break
                elif bb_positive == "best-anchor":
                    scores = iou(anchors[s], np.tile(np.array(box[1:5]), (anchors[s].shape[0], anchors[s].shape[1], 1) ))
                    layer_y, layer_x = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
                    d = ( box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2)**2 + ( box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) **2
                    if grids[s][ i, layer_y, layer_x, num_classes ] == 1.:
                        collisions = collisions + 1
                    if d < grids[s][i, layer_y, layer_x, num_classes + 5] :
                        pos[s] += 1
                        grids[s][ i, layer_y, layer_x, :num_classes ] = 0.0
                        grids[s][ i, layer_y, layer_x, cl ] = 1.0
                        grids[s][ i, layer_y, layer_x, num_classes ] = 1.0
                        grids[s][ i, layer_y, layer_x, num_classes + 1 ] = ( box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2) / dim # y
                        grids[s][ i, layer_y, layer_x, num_classes + 2 ] = ( box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) / dim # x
                        grids[s][ i, layer_y, layer_x, num_classes + 3 ] = math.log(dim/box[3], 2)
                        grids[s][ i, layer_y, layer_x, num_classes + 4 ] = math.log(dim/box[4], 2)
                    if d < grids[s][i, layer_y, layer_x, num_classes + 5]:
                        grids[s][i, layer_y, layer_x, num_classes + 5] = d
                    break
                else:
                    print("Unknown bb_positive argument")


    for s in range(len(layer_sizes)):
        print("{} positives on layer {}".format(pos[s], s))

    print("{} collisions".format(collisions))

    return grids

# second version: implementation with Numpy computations per image

def iou_(a, b):
    a = np.array(a).reshape((-1, 1, 4))
    b = np.array(b).reshape((1, -1, 4))
    a = np.tile(a, (1, b.shape[1], 1) )
    b = np.tile(b, (a.shape[0], 1, 1) )

    intersection = np.maximum((np.minimum(a[..., 0] + a[...,2], b[...,0] + b[...,2]) - np.maximum(a[...,0], b[...,0])), 0) * \
        np.maximum((np.minimum(a[...,1] + a[...,3], b[...,1] + b[...,3]) - np.maximum(a[...,1], b[...,1])), 0)
    union = (np.maximum(a[...,0] + a[...,2], b[...,0] + b[...,2]) - np.minimum(a[...,0], b[...,0])) * \
        (np.maximum(a[...,1] + a[...,3], b[...,1] + b[...,3]) - np.minimum(a[...,1], b[...,1]) )
    return intersection / union

def prio_distance(a, b):
    a = np.array(a).reshape((-1, 1, 4))
    b = np.array(b).reshape((1, -1 ,4))
    a = np.tile(a, (1, b.shape[1], 1) )
    b = np.tile(b, (a.shape[0], 1, 1) )

    return (b[...,0] + b[...,2]/2 - a[...,0] - a[...,2]/2)**2 + (b[...,1] + b[...,3]/2 - a[...,1] - a[...,3]/2)**2

def compute_grids_(start_index, nb_images, groundtruth, layer_offsets, layer_strides, layer_fields, input_shape, stride_margin, iou_treshold, num_classes, verbose=False):
    anchors = get_positioned_anchors(input_shape, layer_offsets, layer_strides, layer_fields, stride_margin)
    layer_sizes = get_layer_sizes(input_shape, layer_offsets, layer_strides)
    layer_attributes = list(zip(layer_offsets, layer_strides, layer_fields, layer_sizes, anchors))

    if verbose:
        print("Computing grids...")
    grids = []
    for _, _, _, si, _ in layer_attributes:
        grids.append( np.zeros((nb_images, si[0], si[1], num_classes+6 ), dtype='float32'))

    previous_dim = 0
    for s, (layer_offset, layer_stride, layer_field, layer_size, anchor) in enumerate(layer_attributes):
        if verbose:
            print("   Layer", s)
        # we consider only letters smaller than this dimension
        # and a dimension smaller than the layer field
        # to ensure that 1) at least one anchor sees the full letter
        # 2) the layer field captures the blank around the letter
        # See margin=True in get_positioned_anchors method
        dim = layer_field
        if stride_margin:
            dim = dim - layer_stride

        # if previous_dim == 0:
        #     previous_dim = dim / 4

        for n in range(nb_images):
            layer_boxes = []
            for b in groundtruth:
                if (b[0] == start_index + n) and (b[4] <= dim) and (b[3] <= dim) and ( max(b[3], b[4]) > previous_dim ) : # and ( min(b[3], b[4]) > 0 ) :
                    layer_boxes.append(b)

            if verbose:
                print("      Nb of chars of size < layer field:", len(layer_boxes))
                print("      IOU threshold:", iou_treshold)
            if len(layer_boxes)>0:
                layer_boxes = np.array(layer_boxes, dtype=np.float32)
                positives = layer_boxes[:, 1:5]
                labels = layer_boxes[:, 5]

                scores = iou_(anchor, positives)
                if verbose:
                    print("      Ground truth with a target:", np.sum(np.amax(scores, axis=0) > iou_treshold))

                scores = scores.reshape((anchor.shape[0], anchor.shape[1], -1)) # (grid-x, grid-y, groundtruth)
                mask = scores > iou_treshold
                mask = mask.astype(int)

                if verbose:
                    print("      Nb collisions (same targets):", np.sum(np.sum(mask, axis=2) - np.max(mask, axis=2)))

                positive_grid_positions = np.max(mask, axis=2) > 0

                distances = prio_distance(anchor, positives)
                distances = distances.reshape((anchor.shape[0], anchor.shape[1], -1)) # (grid-x, grid-y, groundtruth)
                priority_index = np.argmax( - distances + 100000 * mask, axis=2)
                positives_index = priority_index[positive_grid_positions]
                if verbose:
                    print("      Nb of positive positions with sufficient IOU with GT:", len(positives_index))

                min_distances = np.min(distances, axis=2)

                labels = labels[positives_index]
                from keras.utils import to_categorical
                labels = to_categorical(labels, num_classes=num_classes)

                objectness = np.ones(positives_index.shape)

                positives_coords = positives[positives_index]
                positive_anchors = anchor[positive_grid_positions]

                reg_y = ( positives_coords[...,0] + positives_coords[...,2] /2 - positive_anchors[..., 0] - positive_anchors[..., 2]/2) / dim
                reg_x = ( positives_coords[...,1] + positives_coords[...,3] /2 - positive_anchors[..., 1] - positive_anchors[..., 3]/2) / dim
                reg_h = np.log2( dim / positives_coords[...,2])
                reg_w = np.log2( dim / positives_coords[...,3])

                grids[s][n][positive_grid_positions] =  np.concatenate([labels, np.stack((objectness, reg_y, reg_x, reg_h, reg_w, min_distances[positive_grid_positions]), axis=1)], axis=1)
                grids[s][n,:, :, num_classes+5] = min_distances
            else:
                if verbose:
                    print("      0 positives for this layer due to character size.")

        previous_dim = dim

    return grids

# Thrid version: faster, only considering local 4 anchors

def compute_grids_local(start_index, nb_images, groundtruth, layer_offsets, layer_strides, layer_fields, input_shape, stride_margin, iou_treshold, num_classes):
    print("Computing grids...")
    anchors = get_positioned_anchors(input_shape, layer_offsets, layer_strides, layer_fields, stride_margin)
    layer_sizes = get_layer_sizes(input_shape, layer_offsets, layer_strides)

    grids = [] # create different grids for different downsizing factors
    for layer_size in layer_sizes:
        grid = np.zeros( [nb_images] + layer_size + [num_classes+6], dtype = 'float32')
        grid[..., num_classes+5] = 100000
        grids.append(grid)
    collisions = 0
    pos = np.zeros([len(layer_sizes)]) # statistics

    for box in groundtruth:
        i = box[0] - start_index
        cl = box[5]
        for s in range(len(layer_sizes)):
            # we consider only letters smaller than this dimension
            # and a dimension smaller than the layer field
            # to ensure that 1) at least one anchor sees the full letter
            # 2) the layer field captures the blank around the letter
            # See margin=True in get_positioned_anchors method
            dim = layer_fields[s]
            if stride_margin:
                dim = dim - layer_strides[s]

            if (box[4] <= dim) and (box[3] <= dim) :
                # take position from top left corner of the box
                pos_layer_y, pos_layer_x = get_position_on_grid((box[1] + box[3]/2, box[2] + box[4]/2), layer_offset=layer_offsets[s], layer_stride=layer_strides[s], layer_size=layer_sizes[s])

                max_range = math.ceil(dim / layer_strides[s]) -1
                for layer_y in range(max(pos_layer_y - max_range,0), min(pos_layer_y + max_range + 1, layer_sizes[s][0]) ):
                    for layer_x in range(max(pos_layer_x - max_range, 0), min(pos_layer_x + max_range + 1, layer_sizes[s][1]) ):
                        score = iou(anchors[s][layer_y, layer_x], np.array(box[1:5]))
                        d = ( box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2)**2 + ( box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) **2

                        if score > iou_treshold:
                            if grids[s][ i, layer_y, layer_x, num_classes ] == 1.:
                                collisions = collisions + 1
                            if d < grids[s][ i, layer_y, layer_x, num_classes + 5 ] :
                                pos[s] += 1
                                grids[s][ i, layer_y, layer_x, :num_classes ] = 0.0
                                grids[s][ i, layer_y, layer_x, cl ] = 1.0
                                grids[s][ i, layer_y, layer_x, num_classes ] = 1.0
                                grids[s][ i, layer_y, layer_x, num_classes + 1 ] = ( box[1] + box[3]/2 - anchors[s][layer_y, layer_x, 0] - anchors[s][layer_y, layer_x, 2]/2) / dim # y
                                grids[s][ i, layer_y, layer_x, num_classes + 2 ] = ( box[2] + box[4]/2 - anchors[s][layer_y, layer_x, 1] - anchors[s][layer_y, layer_x, 3]/2) / dim # x
                                grids[s][ i, layer_y, layer_x, num_classes + 3 ] = math.log(dim/box[3], 2)
                                grids[s][ i, layer_y, layer_x, num_classes + 4 ] = math.log(dim/box[4], 2)
                        if d < grids[s][ i, layer_y, layer_x, num_classes + 5 ]:
                            grids[s][ i, layer_y, layer_x, num_classes + 5 ] = d
                break


    for s in range(len(layer_sizes)):
        print("{} positives on layer {}".format(pos[s], s))

    print("{} collisions".format(collisions))

    return grids

# from keras.preprocessing.image import ImageDataGenerator
# def get_img_fit_flow(image_config, fit_smpl_size, directory, target_size, batch_size, shuffle):
#     '''
#     Sample the generators to get fit data
#     image_config  dict   holds the vars for data augmentation &
#     fit_smpl_size float  subunit multiplier to get the sample size for normalization
#
#     directory     str    folder of the images
#     target_size   tuple  images processed size
#     batch_size    str
#     shuffle       bool
#     '''
#     if 'featurewise_std_normalization' in image_config and image_config['image_config']:
#        img_gen = ImageDataGenerator()
#        batches = img_gen.flow_from_directory(
#           directory=directory,
#           target_size=target_size,
#           batch_size=batch_size,
#           shuffle=shuffle,
#         )
#        fit_samples = np.array([])
#        fit_samples.resize((0, target_size[0], target_size[1], 3))
#        for i in range(batches.samples/batch_size):
#            imgs, labels = next(batches)
#            idx = np.random.choice(imgs.shape[0], batch_size*fit_smpl_size, replace=False)
#            np.vstack((fit_samples, imgs[idx]))
#     new_img_gen = ImageDataGenerator(**image_config)
#     if 'featurewise_std_normalization' in image_config and image_config['image_config']:
#         new_img_gen.fit(fit_samples)
#     return new_img_gen.flow_from_directory(
#        directory=directory,
#        target_size=target_size,
#        batch_size=batch_size,
#        shuffle=shuffle,
#     )

## Data

# ocr datasets
import datasets.ocr_mnist
import datasets.ocr_documents
import datasets.ocr_documents_generator

# classification datasets
import datasets.cls_rvl_cdip
import datasets.cls_tiny_imagenet
import datasets.cls_dogs_vs_cats



def get(**kwargs):
    if kwargs.get("name") not in globals():
        raise KeyError('Unknown dataset: {}'.format(kwargs))

    return globals()[kwargs.get("name")].Dataset(**kwargs)
