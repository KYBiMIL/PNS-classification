

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, GaussianDropout, Lambda, BatchNormalization
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense, AveragePooling2D
from keras.layers import concatenate, Add, Multiply
from keras.layers import Activation
from keras.layers import ZeroPadding2D, Cropping2D, GlobalAveragePooling2D
from keras import backend as K
from keras.initializers import glorot_uniform


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn

def transition_block(x,f,channel,stage):
	
	bn_namebase = 'batch' + str(stage) + '_branch'
	conv_namebase = 'dense' + str(stage) + '_branch'

	x = BatchNormalization(axis=-1, name=bn_namebase+'_b')(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=channel, kernel_size=(f,f), strides=(1,1), padding='same',
name=conv_namebase+'_b', kernel_initializer=glorot_uniform(seed=1234))(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  

	return x

def dense_block(x,f,channel,stage):


	conv_namebase = 'res' + str(stage) + '_branch'
	bn_namebase = 'bn' + str(stage) + '_branch'
	x_shortcut = x

	x = BatchNormalization(axis=-1, name=bn_namebase+'_a')(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=channel, kernel_size=(f,f), strides=(1,1), padding='same',
name=conv_namebase+'_a', kernel_initializer=glorot_uniform(seed=1234))(x)

	x_shortcut = BatchNormalization(axis=-1, name=bn_namebase+'1')(x_shortcut)
	x_shortcut = Activation('relu')(x_shortcut)	
	x_shortcut = Conv2D(filters=channel, kernel_size=(1, 1), strides=(1, 1), name=conv_namebase+'1', kernel_initializer=glorot_uniform(seed=1234))(x_shortcut)

	x = concatenate([x, x_shortcut],axis=-1)

	return x


def MultiDense_HyunahModel(input_shape_1, num_classes, weights=None):
 
## input_shape_1 is larger than input_shape_0

	if num_classes == 2:
		num_classes = 1
		loss = 'binary_crossentropy'
		activation = 'sigmoid'
	else:
		loss = 'categorical_crossentropy'
		activation = 'softmax'

	data = Input(shape=input_shape_1, dtype='float', name='data')
	x = Lambda(mvn, name='mvn')(data)

	x0 = Conv2D(48,kernel_size=(3,3),strides=(1,1), padding='same')(x)
	x1 = Conv2D(108,kernel_size=(3,3),strides=(1,1), padding='same')(x)
	
	x0 = dense_block(x0, 3, 108, 1)
	x0 = concatenate([x1,x0],axis=-1)

	x0 = transition_block(x0,1,108,2)
	x0 = dense_block(x0,3,168,3)
	
	x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
	x2 = Conv2D(168, kernel_size=(3,3),strides=(1,1), padding='same')(x2)
	x0 = concatenate([x2,x0],axis=-1)

	x0 = transition_block(x0,1,168,4)

	x0 = GlobalAveragePooling2D()(x0)
	x0 = Dense(256, activation='relu', name='FC0')(x0)
	x0 = Dense(96, activation='relu', name='FC1')(x0)

	"""
	x0 = dense_block(x0,3,228,5)
		
	x3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)
	x3 = Conv2D(228, kernel_size=(3,3),strides=(1,1), padding='same')(x3)
	x0 = concatenate([x3,x0],axis=-1)

	x0 = transition_block(x0,1,228,6)

	x0 = GlobalAveragePooling2D()(x0)
	x0 = Dense(256, activation='relu', name='FC0')(x0)
	x0 = Dense(96, activation='relu', name='FC1')(x0)
	"""
	prediction = Dense(num_classes, activation=activation, name='combined_predictions')(x0)
	model = Model(inputs=data, outputs=prediction)
	
	sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.8, nesterov=True)
	#rmsp = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay = 1e-6)
	model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])

	return model

if __name__ == '__main__':
	model = MultiDense_HyunahModel((224, 224, 3), 2, weights=None)



