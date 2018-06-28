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
        self.strides = [32]
        self.offsets = [16]
        self.fields = [224]

    def build(self, input_shape, num_classes):

        assert input_shape == (224, 224, 3) , "incorrect input shape " + input_shape
        image_input = Input(shape=input_shape, name='image_input')

        model = Sequential()
        # block 1
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same', name='block1_conv1')) # 224 x 224
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", name='block1_pool')) # 112x112
        # model.add(Dropout(0.25))

        # block 2
        model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation='relu', name='block2_conv1'))
        model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation='relu', name='block2_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", name='block2_pool')) # 56 x 56
        # model.add(Dropout(0.5))

        # block 3
        model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation='relu', name='block3_conv1'))
        model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation='relu', name='block3_conv2'))
        model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation='relu', name='block3_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", name='block3_pool')) # 28 x 28

        # block 4
        model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', name='block4_conv1'))
        model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', name='block4_conv2'))
        model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', name='block4_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", name='block4_pool')) # 14 x 14

        # block 5
        model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', name='block5_conv1'))
        model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', name='block5_conv2'))
        model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation='relu', name='block5_conv3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid", name='block5_pool')) # 7 x 7

        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax', name='predictions'))

        # GlobalAveragePooling2D()

        self.model = Model(image_input, model(image_input))
        self.model.strides = self.strides
        self.model.offsets = self.offsets
        self.model.fields = self.fields
        return self.model
