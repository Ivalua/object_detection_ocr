from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, concatenate, GlobalAveragePooling2D
from keras import applications

class Network:

    def __init__(self, stride_scale = 0):
        # if stride_scale == 0:
        #     self.stride_scale = 14
        # else:
        #     self.stride_scale = stride_scale
        #
        # self.strides = [2 * self.stride_scale]
        self.strides = [32]
        self.offsets = [16]
        self.fields = [150]

    def build(self, input_shape, num_classes):

        vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        model = Sequential()
        for l in vgg.layers:
            #l.trainable = False
            model.add(l)

        model.add(Conv2D(num_classes, (1, 1)))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))

        print(model.summary())

        image_input = Input(shape=input_shape, name='image_input')

        self.model = Model(image_input, model(image_input))
        self.model.strides = self.strides
        self.model.offsets = self.offsets
        self.model.fields = self.fields
        return self.model
