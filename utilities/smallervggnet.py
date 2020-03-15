# import the necessary packages
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		inputShape = (height, width, depth)
		model = keras.applications.resnet50.ResNet50(include_top=False, #Do not include FC layer at the end
                                                     input_shape=inputShape,
                                                     weights='imagenet')
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
        
		x = model.output

		#Flatten the output to feed to Dense layer
		x = Flatten()(x)

		#Add Dropout
		x = Dropout(0.5)(x)

		#Add one Dense layer
		x = Dense(200, activation='relu')(x)

		#Batch Norm
		x = BatchNormalization()(x)
        

# 		# CONV => RELU => POOL
# 		model.add(Conv2D(32, (3, 3), padding="same",
# 			input_shape=inputShape))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(MaxPooling2D(pool_size=(3, 3)))
# 		model.add(Dropout(0.25))

# 		# (CONV => RELU) * 2 => POOL
# 		model.add(Conv2D(64, (3, 3), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(Conv2D(64, (3, 3), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(MaxPooling2D(pool_size=(2, 2)))
# 		model.add(Dropout(0.25))

# 		# (CONV => RELU) * 2 => POOL
# 		model.add(Conv2D(128, (3, 3), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(Conv2D(128, (3, 3), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(MaxPooling2D(pool_size=(2, 2)))
# 		model.add(Dropout(0.25))
        
# 		# (CONV => RELU) * 2 => POOL
# 		model.add(Conv2D(128, (3, 3), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(Conv2D(128, (3, 3), padding="same"))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization(axis=chanDim))
# 		model.add(MaxPooling2D(pool_size=(2, 2)))
# 		model.add(Dropout(0.25))

# 		# first (and only) set of FC => RELU layers
# 		model.add(Flatten())
# 		model.add(Dense(1024))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization())
# 		model.add(Dropout(0.5))
        
# 		# first (and only) set of FC => RELU layers
# 		model.add(Dense(512))
# 		model.add(Activation("relu"))
# 		model.add(BatchNormalization())
# 		model.add(Dropout(0.5))

# 		# softmax classifier
# 		x = Dense(classes)(x)
# 		x = Activation(finalAct)(x)
        
		label_output = Dense((classes), activation=finalAct, 
                             name='class_op')(x)
        
        #Non Sequential model as it has two different outputs
		final_model = keras.models.Model(inputs=model.input, #Pre-trained model input as input layer
                                            outputs=label_output) #Output layer added

		# return the constructed network architecture
		return final_model