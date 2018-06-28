from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, concatenate

class Network:

    def __init__(self, stride_scale = 0):
        if stride_scale == 0:
            self.stride_scale = 14
        else:
            self.stride_scale = stride_scale

        self.strides = [2 * self.stride_scale, 4 * self.stride_scale]
        self.offsets = [14, 28]
        self.fields = [28, 56]


    def build(self, input_shape, num_classes):
        image_input = Input(shape=input_shape, name='image_input')

        # stage 1
        model = Sequential() # 28, 28, 1
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=input_shape, padding='valid')) # 28, 28, 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid')) # 28, 28, 1
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid")) # 14, 14, 1
        s1 = model(image_input)

        # stage 2
        s2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid')(s1)
        s2 = Conv2D(64, (3, 3), activation='relu', padding='valid')(s2)
        s2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid")(s2)

        # output 1
        f1 = Dropout(0.25)(s1)
        f1 = Conv2D(128, kernel_size=(12,12), strides=(self.stride_scale,self.stride_scale),
                                            padding="valid", activation='relu')(f1)
        f1 = Dropout(0.5)(f1)
        output1_1 = Dense(num_classes, activation = "softmax")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f1)))
        output1_2 = Dense(1, activation = "sigmoid")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f1)))
        output1_3 = Dense(2, activation = "tanh")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f1)))
        output1_4 = Dense(2, activation = "sigmoid")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f1)))

        output1 = concatenate([output1_1, output1_2, output1_3, output1_4])

        # output 2
        f2 = Dropout(0.25)(s2)
        f2 = Conv2D(128, kernel_size=(11,11), strides=(self.stride_scale,self.stride_scale),
                                            padding="valid", activation='relu')(f2)
        f2 = Dropout(0.5)(f2)
        output2_1 = Dense(num_classes, activation = "softmax")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f2)))
        output2_2 = Dense(1, activation = "sigmoid")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f2)))
        output2_3 = Dense(2, activation = "tanh")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f2)))
        output2_4 = Dense(2, activation = "sigmoid")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(f2)))

        output2 = concatenate([output2_1, output2_2, output2_3, output2_4])

        # print(model.summary())
        # if K._BACKEND=='tensorflow':
        # 	for layer in model.layers:
        # 		print(layer.get_output_at(0).get_shape().as_list())
        self.model = Model(image_input, [output1, output2])
        self.model.strides = self.strides
        self.model.offsets = self.offsets
        self.model.fields = self.fields
        return self.model
