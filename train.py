from __future__ import print_function
import os
import time
import datetime
import argparse
from utils import check_config
check_config()
import models, datasets

import keras
from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch_size", type=int,
	default=3, help="# of images per batch")
ap.add_argument("-p", "--parallel", default=False,
	help="Enable multi GPUs", action='store_true')
ap.add_argument("-e", "--epochs", type=int, default=12,
	help="# of training epochs")
ap.add_argument("-l", "--logs", type=str, default="logs",
	help="log directory")
ap.add_argument("-m", "--model", type=str,
	default="CNN_C32_C64_M2_C128_D", help="model")
ap.add_argument("-lr", "--learning_rate", type=float,
	default=0.001, help="learning rate")
ap.add_argument("-s", "--stride_scale", type=int,
	default=0, help="Stride scale. If zero, default stride scale.")
ap.add_argument("-d", "--dataset", type=str, default="ocr_mnist",
	help="dataset")
ap.add_argument("-w", "--white", type=float, default=0.9,
	help="white probability for MNIST dataset")
ap.add_argument("-n", "--noise", default=False,
	help="noise for MNIST dataset", action='store_true')
ap.add_argument("--pos_weight", type=float, default=100.,
	help="weight for positive objects")
ap.add_argument("--iou", type=float, default=.3,
	help="iou treshold to consider a position to be positive. If -1, positive only if \
	object included in the layer field")
ap.add_argument("--bb_positive", type=str, default="iou-treshold",
	help="Possible values: iou-treshold, in-anchor, best-anchor")
ap.add_argument("--nms_iou", type=float, default=.2,
	help="iou treshold for non max suppression")
ap.add_argument("-i", "--input_dim", type=int, default=700,
	help="network input dim")
ap.add_argument("-r", "--resize", type=str, default="",
	help="resize input images")
ap.add_argument('--no-save', dest='save', action='store_false',
	help="save model and data to files")
ap.add_argument('--resume', dest='resume_model', type=str, default="")
ap.add_argument('--n_cpu', type=int, default=1,
	help='number of CPU threads to use during data generation')
args = ap.parse_args()
print(args)

assert K.image_data_format() == 'channels_last' , "image data format channel_last"

model = models.get(name = args.model, stride_scale = args.stride_scale)
print("#"*14 +" MODEL "+ "#"*14)
print("### Stride scale:              " + str(model.stride_scale))
for s in model.strides: print("### Stride:                    " + str(s))
print("#" * 35)

dataset = datasets.get(name = args.dataset, layer_strides = model.strides, layer_offsets = model.offsets,
						layer_fields = model.fields, white_prob = args.white, bb_positive = args.bb_positive, iou_treshold=args.iou, save=args.save,
						batch_size = args.batch_size, input_dim=args.input_dim, resize=args.resize, noise=args.noise)

# model initialization and parallel computing
if not args.parallel:
	print("[INFO] training with 1 device...")
	built_model = model.build(input_shape = dataset.input_shape, num_classes = dataset.num_classes)
else:
	if K._BACKEND=='tensorflow':
		from tensorflow.python.client import device_lib
		def get_available_gpus():
			local_device_protos = device_lib.list_local_devices()
			return [x.name for x in local_device_protos if x.device_type == 'GPU']
		ngpus = len(get_available_gpus())
		print("[INFO] training with {} GPUs...".format(ngpus))
		import tensorflow as tf
		with tf.device("/cpu:0"):
			original_built_model = model.build(input_shape = dataset.input_shape, num_classes= dataset.num_classes)
		built_model = multi_gpu_model(original_built_model, gpus=ngpus)
	elif K._BACKEND=='cntk':
		built_model = model.build(input_shape = dataset.input_shape, num_classes= dataset.num_classes)
	else:
		print("Multi GPU not available on this backend.")

# import numpy as np
# class_weights = np.ones(dataset.num_classes)

# model compilation with loss and accuracy
def custom_loss(y_true, y_pred):
	final_loss = 0.
	if dataset.enable_boundingbox:
		obj_true = y_true[...,dataset.num_classes]
		obj_pred = y_pred[...,dataset.num_classes]
		#   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
		log_weight = 1. + (args.pos_weight - 1.) * obj_true
		obj = (1. - obj_true) * obj_pred + log_weight * (K.log(1. + K.exp(-K.abs(obj_pred))) + K.relu(- obj_pred))

		obj = K.square(obj_pred - obj_true)

		prob = y_pred[...,0:dataset.num_classes]
		# scale predictions so that the class probas of each sample sum to 1
		prob /= K.sum(prob, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		prob = K.clip(prob, K.epsilon(), 1 - K.epsilon())
		# calc
		loss = y_true[...,0:dataset.num_classes] * K.log(prob) #* class_weights
		cat = -K.sum(loss, -1, keepdims=True)

		reg = K.sum(K.square(y_true[..., dataset.num_classes+1:dataset.num_classes+5] - y_pred[...,dataset.num_classes+1:dataset.num_classes+5]), axis=-1, keepdims=True)

		# if args.best_position_classification:
		# 	mask = K.cast( K.less_equal( y_true[..., dataset.num_classes+5:(dataset.num_classes+6)], model.strides[0] * 1.42 / 2  ), K.floatx())

		mask = K.cast( K.equal( y_true[..., dataset.num_classes:(dataset.num_classes+1)], 1.0  ), K.floatx())

		final_loss = final_loss + obj + K.sum(cat * mask) / K.maximum(K.sum(mask), 1.0) + 100 * K.sum(reg * mask) / K.maximum(K.sum(mask), 1.0)

	if dataset.enable_classification or dataset.enable_segmentation:
        	final_loss = final_loss + K.categorical_crossentropy(y_true, y_pred)

	return final_loss


# metrics
metrics = []
if dataset.enable_boundingbox:

	def obj_accuracy(y_true, y_pred):
		acc = K.cast(K.equal( y_true[...,dataset.num_classes], K.round(y_pred[...,dataset.num_classes])), K.floatx())
		return K.mean(acc)
	metrics.append(obj_accuracy)

	def class_accuracy(y_true, y_pred):
		mask = K.cast( K.equal(y_true[...,dataset.num_classes], 1.0  ), K.floatx()  )
		acc = K.cast(K.equal(K.argmax(y_true[...,0:dataset.num_classes], axis=-1), K.argmax(y_pred[...,0:dataset.num_classes], axis=-1)), K.floatx())
		if K.backend() == "cntk":
			acc = K.expand_dims(acc)
		return K.sum(acc * mask) / K.maximum(K.sum(mask), 1.0)
	metrics.append(class_accuracy)

	def reg_accuracy(y_true, y_pred):
		mask = K.cast( K.equal(y_true[...,dataset.num_classes], 1.0  ), K.floatx()  )
		reg = K.sum(K.square(y_true[...,dataset.num_classes+1:dataset.num_classes+3] - y_pred[...,dataset.num_classes+1:dataset.num_classes+3]), axis=-1)
		if K.backend() == "cntk":
			reg = K.expand_dims(reg)
		return K.sum(reg * mask) / K.maximum(K.sum(mask), 1.0)
	metrics.append(reg_accuracy)

if dataset.enable_classification or dataset.enable_segmentation:
	metrics.append(categorical_accuracy)

# model compilation
# built_model.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(lr=args.learning_rate), metrics=metrics)
built_model.compile(optimizer=keras.optimizers.Adam(lr=args.learning_rate), loss=custom_loss, metrics=metrics)

if args.resume_model:
	print("Resuming model from weights in " + args.resume_model)
	built_model.load_weights(args.resume_model, by_name=True)
# plot_model(built_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# parallel computing on CNTK
if args.parallel and (K._BACKEND=='cntk'):
	import cntk as C
	built_model._make_train_function()
	trainer = built_model.train_function.trainer
	assert (trainer is not None), "Cannot find a trainer in Keras Model!"
	learner_no = len(trainer.parameter_learners)
	assert (learner_no > 0), "No learner in the trainer."
	if(learner_no > 1):
	    warnings.warn("Unexpected multiple learners in a trainer.")
	learner = trainer.parameter_learners[0]
	dist_learner = C.train.distributed.data_parallel_distributed_learner(learner, \
	                     num_quantization_bits=32, distributed_after=0)
	built_model.train_function.trainer = C.trainer.Trainer(
	    trainer.model, [trainer.loss_function, trainer.evaluation_function], [dist_learner])
	rank = C.Communicator.rank()
	workers = C.Communicator.num_workers()
	print("[INFO] CNTK training with {} GPUs...".format(workers))
	total_items = dataset.x_train.shape[0]
	start = rank * total_items//workers
	end = min((rank+1) * total_items // workers, total_items)
	x_train, y_train = dataset.x_train[start : end], dataset.y_train[start : end]


start_time = time.time()

# Callbacks: save and tensorboard display
callbacks = []

if args.save:
	from keras.callbacks import ModelCheckpoint
	if not os.path.exists("/sharedfiles/models"):
		os.makedirs("/sharedfiles/models")
	fname = "/sharedfiles/models/" + datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M_') + args.model + ".h5"
	if args.parallel: # http://github.com/keras-team/keras/issues/8649
		from callback import ParallelSaveCallback
		checkpoint = ParallelSaveCallback(original_built_model,fname)
	else:
		if dataset.enable_boundingbox:
			checkpoint = ModelCheckpoint(fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		else:
			checkpoint = ModelCheckpoint(fname, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
	callbacks.append(checkpoint)

if K._BACKEND=='tensorflow':
	from callback import TensorBoard
	log_dir = './Graph/' + time.strftime("%Y-%m-%d_%H:%M:%S")
	tensorboard = TensorBoard(dataset.gt_test, dataset.classes, dataset.stride_margin, model.strides, model.offsets, model.fields, args.nms_iou,
	    log_dir=log_dir,
	    histogram_freq=0,
		batch_size=args.batch_size,
		max_validation_size=100,
		write_output_images=True,
		enable_segmentation=dataset.enable_segmentation,
		enable_boundingbox=dataset.enable_boundingbox,
	    write_graph=False,
	    write_images=False,
		val_data=dataset.val if  hasattr(dataset,"val") else None
	)
	print("Log saved in ", log_dir)
	tensorboard.set_model(built_model)
	callbacks.append(tensorboard)

# training section
if hasattr(dataset, "x_train"):
	built_model.fit(dataset.x_train, dataset.y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(dataset.x_test, dataset.y_test),
          callbacks=callbacks)
else:
	built_model.fit_generator(dataset.train,
                    epochs=args.epochs,
					verbose=1,
					workers=args.n_cpu,
                    use_multiprocessing=False,
					max_queue_size=10,
					shuffle=True,
                    validation_data=dataset.val,
					callbacks=callbacks)

# # save model
# if args.save:
# 	if not os.path.exists("/sharedfiles/models"):
# 		os.makedirs("/sharedfiles/models")
# 	fname = "/sharedfiles/models/" + datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M_') + args.model + ".h5"
# 	built_model.save_weights(fname)
# 	print("Model weights saved in " + fname)

# evaluate section
if hasattr(dataset, "x_test"):
	score = built_model.evaluate(dataset.x_test, dataset.y_test, batch_size=args.batch_size, verbose=0)
else:
	score = built_model.evaluate_generator(dataset.test)

print("Test loss and accuracy values:")
for s in score:
	print(s)
duration = time.time() - start_time
print('Total Duration (%.3f sec)' % duration)

if K._BACKEND=='tensorflow':
	print("Log saved in ", log_dir)

if K._BACKEND=='cntk' and args.save:
	import cntk as C
	C.combine(built_model.outputs).save(fname[:-3]+'.dnn')

if K._BACKEND=='cntk' and  args.parallel:
	C.Communicator.finalize()
