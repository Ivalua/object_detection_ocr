from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, concatenate

class Network:

    def __init__(self, stride_scale = 0):
        if stride_scale == 0:
            self.stride_scale = 6
        else:
            self.stride_scale = stride_scale

        self.strides = [self.stride_scale]
        self.offsets = [14]
        self.fields = [28]


    def build(self, input_shape, num_classes):
        image_input = Input(shape=input_shape, name='image_input')

        model = Sequential() # 28, 28, 1
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=input_shape, padding='valid')) # 28, 28, 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid')) # 28, 28, 1
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid')) # 28, 28, 1
        model.add(Conv2D(64, (5, 5), activation='relu', dilation_rate=(2, 2), padding='valid')) # 28, 28, 1
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(14,14), strides=(self.stride_scale,self.stride_scale),
                                 padding="valid", activation='relu'))
        model.add(Dropout(0.5))

        features = model(image_input)

        output1 = Dense(num_classes, activation = "softmax")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(features)))
        output2 = Dense(1, activation = "sigmoid")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(features)))
        output3 = Dense(2, activation = "tanh")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(features)))
        output4 = Dense(2, activation = "sigmoid")(Dense(64, activation = "relu")(Dense(128, activation = "relu")(features)))

        output = concatenate([output1, output2, output3, output4], name="output")

        # print(model.summary())
        # if K._BACKEND=='tensorflow':
        # 	for layer in model.layers:
        # 		print(layer.get_output_at(0).get_shape().as_list())
        self.model = Model(image_input, output)
        self.model.strides = self.strides
        self.model.offsets = self.offsets
        self.model.fields = self.fields
        return self.model
