from __future__ import division
from keras.callbacks import Callback
from keras import backend as K
from datasets import get_layer_sizes, iou
import numpy as np
import cv2
import math

colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
          (151, 0, 255), (243, 223, 48), (0, 117, 255),\
          (58, 184, 14), (86, 67, 140), (121, 82, 6),\
          (174, 29, 128), (115, 154, 81), (86, 255, 234)]
np_colors=np.array(colors)

def compute_eligible_rectangles(output_maps, layer_strides, layer_offsets, layer_fields, stride_margin, num_classes, layer_sizes):
    res = []
    for i in range(output_maps[0].shape[0]):
        eligible = []
        for o , output_map in enumerate(output_maps):
            dim = layer_fields[o]
            if stride_margin:
                dim = dim - layer_strides[o]

            # class_prob_map = output_map[0, :, :, 0:nb_classes ] # (15, 25, nb_classes)
            # class_map = np.argmax(class_prob_map, axis=-1) # (15, 25)
            objectness_map = output_map[i, :, :, num_classes ] # (15, 25)
            reg = output_map[i, :, :, num_classes+1:num_classes+5]

            for y in range(objectness_map.shape[0]):
                for x in range(objectness_map.shape[1]):
                    if objectness_map[y, x] > 0.5:
                        w_2 = int(dim * 2**(-reg[y,x,3] -1)) # half width
                        h_2 = int(dim * 2**(-reg[y,x,2] -1)) # half height
                        x1 = layer_offsets[o] + x * layer_strides[o] + reg[y,x,1] * dim - w_2
                        y1 = layer_offsets[o] + y * layer_strides[o] + reg[y,x,0] * dim - h_2
                        x2 = layer_offsets[o] + x * layer_strides[o] + reg[y,x,1] * dim + w_2
                        y2 = layer_offsets[o] + y * layer_strides[o] + reg[y,x,0] * dim + h_2
                        eligible.append( [objectness_map[y, x], y1, x1, 2 * h_2 , 2 * w_2, o ] )
        res.append(eligible)
    return res


def non_max_suppression(rectangles, nms_iou):
    res = []
    for eligible in rectangles:
        valid = []
        if len(eligible) > 0:
            index = np.argsort(- np.array(eligible)[:,0])
            valid.append(eligible[0])
            for i in index:
                if np.max(iou( np.array(valid)[:,1:], np.array( [eligible[i][1:5]] ))) > nms_iou:
                    continue
                else:
                    valid.append( eligible[i] )
        res.append(valid)
    return res


def compute_map_score_and_mean_distance(val_gt, detections, overlap_threshold = 0.5):
    precision_recall = []
    fp, tp = 0, 0
    distance = 0.0

    # unflatten groundtruth, flatten detections for ordering
    nb_groundtruth = 0
    groundtruth = []
    gt_detected = []
    flattened_detections = []
    for image_id in range(len(detections)):
        gt = []
        for r in val_gt:
            if r[0] == image_id:
                gt.append( r[1:5]  )
                nb_groundtruth = nb_groundtruth + 1
        groundtruth.append(gt)
        gt_detected.append(np.zeros((len(gt))))
        for d in range(len(detections[image_id])):
            flattened_detections.append( (detections[image_id][d][0], image_id, d ) )

    # order detections
    if len(flattened_detections) > 0:
        index = np.argsort(- np.array(flattened_detections)[:,0])

        # compute recall and precision for increasingly large subset of detections
        for i in index: # iterate through all predictions
            image_id = flattened_detections[i][1]
            d = flattened_detections[i][2]
            detection = np.array([ detections[image_id][d][1:5] ])

            gt = np.array(groundtruth[image_id])
            if len(gt) == 0:
                fp = fp + 1
            else:
                iou_scores = iou(gt, detection)
                m = np.argmax(iou_scores)
                if iou_scores[m] > overlap_threshold:
                    if gt_detected[image_id][m] == 0: # not yet detected
                        gt_detected[image_id][m] = 1
                        tp = tp + 1
                        distance = distance + math.sqrt(  (gt[m][0] + gt[m][2]/2 - detection[0][0] - detection[0][2]/2)**2 + (gt[m][1] + gt[m][3]/2 - detection[0][1] - detection[0][3]/2)**2  )
                    else: # detected twice
                        fp = fp + 1
                else:
                    fp = fp + 1
            precision_recall.append( (  tp/max(tp+fp, 1), tp/max(nb_groundtruth,1)   ) )

    # filling the dips
    interpolated_precision_recall = []
    for i in range(len(precision_recall)):
        if precision_recall[i][0] >= max( [ p for p, _ in precision_recall[i:] ]  ):
            interpolated_precision_recall.append(precision_recall[i])

    mAP = 0
    previous_r = 0
    for p, r in interpolated_precision_recall:
        mAP += p * (r - previous_r)
        previous_r = r

    return mAP, distance / max(tp, 1)



class TensorBoard(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, val_gt, classes, stride_margin, layer_strides, layer_offsets, layer_fields, nms_iou = .5,
                log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 max_validation_size=10000,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 write_output_images=False,
                 enable_boundingbox=False,
                 enable_segmentation=False,
                 batch_display_freq=100,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None, val_data=None):
        super(TensorBoard, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        self.val_gt = val_gt
        self.classes = classes
        self.num_classes = len(classes)
        self.stride_margin = stride_margin
        self.layer_strides = layer_strides
        self.layer_offsets = layer_offsets
        self.layer_fields = layer_fields
        self.epoch = 0
        self.nms_iou = nms_iou
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_output_images = write_output_images
        self.enable_boundingbox = enable_boundingbox
        self.enable_segmentation = enable_segmentation
        self.batch_display_freq = batch_display_freq
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.max_validation_size = max_validation_size
        self.val_data = val_data

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

        if self.write_output_images:
            self.log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
            self.log_image_name = tf.placeholder(tf.string)
            from tensorflow.python.ops import gen_logging_ops
            from tensorflow.python.framework import ops as _ops
            self.log_image = gen_logging_ops._image_summary(self.log_image_name, tf.expand_dims(self.log_image_data, 0), max_images=1)
            _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, self.log_image)

        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
                        else:
                            # not possible to handle 3D convnets etc.
                            continue

                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            self.saver = tf.train.Saver(list(embeddings.values()))

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_ckpt_path = os.path.join(self.log_dir,
                                                     'keras_embedding.ckpt')

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        if batch % self.batch_display_freq != 0:
            return

        logs = logs or {}
        batch_size = logs.get('size', 0)
        # print(self.model.output.shape)
        # for layer in self.model.layers:
        #     print(layer.name)
        # print(self.model.output.name)

        # self.infer = K.function([self.model.input]+ [K.learning_phase()], [self.model.output] )
        # start_batch = batch * batch_size
        # output_map = self.infer([self.train_data[start_batch:(start_batch+1)], 1]) # [Tensor((1, 25, 15, nb_classes + 1))]

            # self.writer.flush()


    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        logs = logs or {}

        if not self.validation_data:
            # creating validation data from validation generator
            print("Feeding callback validation data with Generator")
            j = 0
            imgs = []
            tags = [[] for s in range(len(self.layer_offsets))]
            for i in self.val_data:
                imgs.append(i[0])
                for s in range(len(self.layer_offsets)):
                    tags[s].append(i[1][s] )
                j = j + 1
                if j > 10:
                    break

            np_imgs = np.concatenate(imgs, axis=0)
            np_tags = []
            for s in range(len(self.layer_offsets)):
                np_tags.append(  np.concatenate( tags[s], axis=0 ) )
            self.validation_data = [np_imgs] + np_tags + [ np.ones(np_imgs.shape[0]), 0.0]

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)


        if self.validation_data and self.write_output_images:
            ######### original image
            # from skimage.io import imsave
            # import os
            # import numpy as np
            # if not os.path.exists(self.log_dir):
            #     os.mkdir(self.log_dir)
            val_img_data = self.validation_data[0]
            val_size = min(val_img_data.shape[0], self.max_validation_size)
            tensors = (self.model.inputs)
            img_shape = val_img_data[0].shape
            layer_sizes = get_layer_sizes(img_shape, self.layer_offsets, self.layer_strides)
            detections, target_detections = [], []
            i = 0
            while i < val_size:
                step = min(self.batch_size, val_size - i)
                batch_val = [val_img_data[i:i + step], 1]
                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]
                    batch_val.append(1)
                feed_dict = dict(zip(tensors, batch_val))

                if self.enable_boundingbox or self.enable_segmentation:
                    output_maps = self.sess.run(self.model.outputs, feed_dict=feed_dict)

                if self.enable_boundingbox:
                    eligible = compute_eligible_rectangles(output_maps,
                                        self.layer_strides, self.layer_offsets, self.layer_fields,
                                        self.stride_margin, self.num_classes, layer_sizes)
                    valid = non_max_suppression(eligible, self.nms_iou)

                    # compute targets for display
                    target = compute_eligible_rectangles([self.validation_data[s+1][i:i+step] for s in range(len(layer_sizes))],
                                        self.layer_strides, self.layer_offsets, self.layer_fields,
                                        self.stride_margin, self.num_classes, layer_sizes)

                if i <= 10:
                    # display results on a few images
                    for image_id in range(step):
                        image = (val_img_data[i + image_id] * 255.)#.astype(np.uint8)
                        if image.shape[2] == 1:
                            image = np.tile(image,(1,1,3))
                        # imsave(os.path.join(self.log_dir, str(image_id) + '_input.png'), image)
                        t = (self.epoch - 1) * val_img_data.shape[0] + i + image_id

                        log_image_summary_op = self.sess.run(self.log_image, \
                                feed_dict={self.log_image_name: "1-input", self.log_image_data: image})
                        self.writer.add_summary(log_image_summary_op, global_step=t)

                        if self.enable_boundingbox:
                            # draw objectness
                            image_ = np.copy(image)
                            for o, output_map in enumerate(output_maps):
                                objectness = output_map[image_id, :, :, self.num_classes: self.num_classes+1 ] * 255.
                                log_image_summary_op = self.sess.run(self.log_image, \
                                        feed_dict={self.log_image_name: "2-objectness-" + str(o), self.log_image_data: np.tile( objectness,(1,1,3)) })
                                self.writer.add_summary(log_image_summary_op, global_step=t)
                                dim = self.layer_fields[o]
                                cv2.rectangle(image_, (0, 0), (dim, dim), colors[o % len(colors)], 2)
                                if self.stride_margin:
                                    dim = dim - self.layer_strides[o]
                                    cv2.rectangle(image_, (0, 0), (dim, dim), colors[o % len(colors)], 2)

                            # draw eligible rectangles (before non max suppression)
                            for r in eligible[image_id]:
                                cv2.rectangle(image_, (int(r[2]), int(r[1])), (int(r[2]+r[4]), int(r[1]+r[3])), colors[r[5] % len(colors)], 2)
                            log_image_summary_op = self.sess.run(self.log_image, \
                                    feed_dict={self.log_image_name: "3-result", self.log_image_data: image_})
                            self.writer.add_summary(log_image_summary_op, global_step=t)

                            # display results (after non max suppression)
                            res_image = np.copy(image)
                            for r in valid[image_id]:
                                cv2.rectangle(res_image, (int(r[2]), int(r[1])), (int(r[2] + r[4]), int(r[1] + r[3])), colors[0], 2)

                            log_image_summary_op = self.sess.run(self.log_image, \
                                    feed_dict={self.log_image_name: "4-after-nms", self.log_image_data: res_image})
                            self.writer.add_summary(log_image_summary_op, global_step=t)

                            # display target label
                            target_image = np.copy(image)
                            for r in target[image_id]:
                                cv2.rectangle(target_image, (int(r[2]), int(r[1])), (int(r[2]+r[4]), int(r[1]+r[3])), colors[r[5] % len(colors)], 2)
                            log_image_summary_op = self.sess.run(self.log_image, \
                                    feed_dict={self.log_image_name: "5-target", self.log_image_data: target_image})
                            self.writer.add_summary(log_image_summary_op, global_step=t)

                            # display groundtruth boxes
                            for r in self.val_gt:
                                if r[0] == i + image_id:
                                    cv2.rectangle(image, (int(r[2]), int(r[1])), (int(r[2]+r[4]), int(r[1]+r[3])), (86 / 255., 0, 240/255.), 2)
                            log_image_summary_op = self.sess.run(self.log_image, \
                                    feed_dict={self.log_image_name: "6-groundtruth", self.log_image_data: image})
                            self.writer.add_summary(log_image_summary_op, global_step=t)

                        if self.enable_segmentation:
                            # draw segmentation maps
                            for o, output_map in enumerate(output_maps):
                                output_map_labels = np.argmax( output_map[image_id], axis=-1 )
                                output_map_color = np_colors[output_map_labels]
                                log_image_summary_op = self.sess.run(self.log_image, \
                                        feed_dict={self.log_image_name: "2-segmentation-" + str(o), self.log_image_data: output_map_color })
                                self.writer.add_summary(log_image_summary_op, global_step=t)

                                target_image = np.copy(image)
                                target_map_labels = np.argmax(self.validation_data[o+1][i+image_id], axis=-1)
                                target_map_color = np_colors[target_map_labels]
                                log_image_summary_op = self.sess.run(self.log_image, \
                                        feed_dict={self.log_image_name: "3-target-segmentation-" + str(o), self.log_image_data: target_map_color  }) # cv2.addWeighted(target_image,0.1,cv2.resize(target_map_color, (target_image.shape[1], target_image.shape[0])),0.9,0, dtype=cv2.CV_32F)
                                self.writer.add_summary(log_image_summary_op, global_step=t)

                # next batch
                if self.enable_boundingbox:
                    detections = detections + valid
                    target_detections = target_detections + non_max_suppression(target, self.nms_iou)
                i += step

            # compute statistics on full val dataset
            if self.enable_boundingbox:
                # mAP score and mean distance
                map, mean_distance = compute_map_score_and_mean_distance(self.val_gt, detections)
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = map
                summary_value.tag = "validation_average_precision"
                self.writer.add_summary(summary, global_step=self.epoch)

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = mean_distance
                summary_value.tag = "validation_mean_distance"
                self.writer.add_summary(summary, global_step=self.epoch)

                # target mAP score
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value, _ = compute_map_score_and_mean_distance(self.val_gt, target_detections)
                summary_value.tag = "target_average_precision"
                self.writer.add_summary(summary, global_step=self.epoch)

        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()


class ParallelSaveCallback(Callback):

    def __init__(self, model, file):
         self.model_to_save = model
         self.file_path = file

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(self.file_path + '_%d.h5' % epoch)
