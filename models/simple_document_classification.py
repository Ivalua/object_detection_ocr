from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, concatenate

class Network:

    def __init__(self, stride_scale = 0):
        # if stride_scale == 0:
        #     self.stride_scale = 14
        # else:
        #     self.stride_scale = stride_scale
        #
        # self.strides = [2 * self.stride_scale]
        self.strides = [0]
        self.offsets = [ ( 5 - 1) / 2.0 ]
        self.fields = [150]

    def build(self, input_shape, num_classes):

        assert input_shape == (150, 150, 3) , "incorrect input shape " + input_shape
        image_input = Input(shape=input_shape, name='image_input')

        model = Sequential()
        # Convolution + Pooling Layer
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Convolution + Pooling Layer
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Convolution + Pooling Layer
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Convolution + Pooling Layer
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flattening
        model.add(Flatten())
        # Fully connection
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.6))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.3))
        model.add(Dense(num_classes, activation='softmax', name='predictions'))

        # GlobalAveragePooling2D()

        self.model = Model(image_input, model(image_input))
        self.model.strides = self.strides
        self.model.offsets = self.offsets
        self.model.fields = self.fields
        return self.model
