# kfmodels.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import InceptionResNetV2

#################################################################
# Model : KFDummy
# KF 12/10/2018
#################################################################
class KFDummy:
    @staticmethod
    def build(input_shape, label_dim=17):
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(label_dim, activation='softmax')
            ])

        return model 


#################################################################
# Model : KFSmallerVGGNet
# KF 12/10/2018
#################################################################
class KFSmallerVGGNet:
    @staticmethod
    def build(input_shape, label_dim=17):

        factor = 4
        # Build small vgg model from scratch
        model = Sequential()
        chanDim = -1
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64*factor, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64*factor, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128*factor, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128*factor, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(256*factor, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256*factor, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())

        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(label_dim))
        model.add(Activation("softmax"))

        return model

 
################################################################# # Model : InceptionResnetV2
# KF 12/19/2018
#################################################################
class KFInceptionResNetV2:
    @staticmethod
    def build(input_shape, label_dim=17):
        # Get convolutional layers
        conv = InceptionResNetV2(
                        weights='imagenet',
                        include_top=False,
                        input_shape=input_shape
                        )
        # Initialize model
        model = Sequential()
        model.add(conv)
        # Add new layers
        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(cls_number, activation='softmax'))

        return model 
