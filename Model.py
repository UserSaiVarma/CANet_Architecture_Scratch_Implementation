###################################################################################################################
#                                           Importing Libraries                                                   #
###################################################################################################################
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

###################################################################################################################
#                                     Convolutional and Identity Block                                            #
###################################################################################################################
class convolutional_block(tf.keras.layers.Layer):
    "Building reusable convolutional block"
    def __init__(self, kernel=3, filters=[4,4,8], stride=1, name="conv_block"):
        super().__init__(name=name)
        self.F1, self.F2, self.F3 = filters
        self.kernel = kernel
        self.stride = stride
        
        #all the convolutional blocks
        self.conv1 = Conv2D(filters=self.F1, kernel_size=(1, 1), strides=(self.stride, self.stride), padding='same')
        self.conv2 = Conv2D(filters=self.F2, kernel_size=(self.kernel, self.kernel), strides=(1,1), padding='same')
        self.conv3 = Conv2D(filters=self.F3, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.convpar = Conv2D(filters=self.F3, kernel_size=(self.kernel, self.kernel), strides=(self.stride, self.stride), padding='same')

        #All batch normalizations
        self.bn1 = BatchNormalization(axis=3)
        self.bn2 = BatchNormalization(axis=3)
        self.bn3 = BatchNormalization(axis=3)
        self.bnpar = BatchNormalization(axis=3)

        #Activation block
        self.activation = Activation('relu')

        #element wise sum
        self.add = Add()
    
    def call(self, X):
        "Architecture of Convolutional block"
        #x_ip = X.copy()
        #first conv  block
        conv1 = self.conv1(X)
        bn1 = self.bn1(conv1)
        activ1 = self.activation(bn1)

        #second conv block
        conv2 = self.conv2(activ1)
        bn2 = self.bn2(conv2)
        activ2 = self.activation(bn2)

        #thrid conv block
        conv3 = self.conv3(activ2)
        bn3 = self.bn3(conv3)

        #parllel conv block
        conv_par = self.convpar(X)
        bn_par = self.bnpar(conv_par)
        activ_par = self.activation(bn_par)

        #adding
        X = Add()([bn3, activ_par])

        #final activation
        X = self.activation(X)

        return X

class identity_block(tf.keras.layers.Layer):
    "Building reusable Identity Block"
    def __init__(self, kernel=3,  filters=[4,4,8], name="identity_block"):
        super().__init__(name=name)
        self.F1, self.F2, self.F3 = filters
        self.kernel = kernel

        #convolutional blocks
        self.conv1 = Conv2D(filters=self.F1, kernel_size=(1,1), strides=(1,1), padding='same')
        self.conv2 = Conv2D(filters=self.F2, kernel_size=(self.kernel, self.kernel), strides=(1,1), padding='same')
        self.conv3 = Conv2D(filters=self.F3, kernel_size=(1,1), strides=(1,1), padding='same')

        #Batch Normalizaiton
        self.bn1 = BatchNormalization(axis=3)
        self.bn2 = BatchNormalization(axis=3)
        self.bn3 = BatchNormalization(axis=3)

        #activation
        self.activation = Activation("relu")

        #element wise addition
        self.add = Add()

    def call(self, X):
        
        #first conv block
        conv1 = self.conv1(X)
        bn1 = self.bn1(conv1)
        activ1 = self.activation(bn1)

        #second conv block
        conv2 = self.conv2(activ1)
        bn2 = self.bn2(conv2)
        activ2 =  self.activation(bn2)

        #third conv block
        conv3 = self.conv3(activ2)
        bn3 = self.bn3(conv3)
        
        #addition
        X = self.add([bn3, X])

        #final activation
        X = self.activation(X)

        return X

###################################################################################################################
#                                        Global and Context flow                                                  #
###################################################################################################################
class global_flow(tf.keras.layers.Layer):
    "Global Flow block"
    def __init__(self, width, height, channels, name="global_flow"):
        super().__init__(name=name)
        self.width = width
        self.height = height
        self.channels = channels

        #necessary blocks
        self.conv1 = Conv2D(filters=self.channels, kernel_size=(1,1), strides=(1,1), padding='same')
        self.global_avg_pool = GlobalAveragePooling2D()
        self.bn = BatchNormalization(axis=3)
        self.activation = Activation("relu")
        self.upsample = UpSampling2D(size=(self.width, self.height), interpolation='bilinear')

    def call(self, X):
        #applying global pooling on the input
        glob_avg = self.global_avg_pool(X)

        #reshaping the size of image after pooling
        glob_avg = tf.expand_dims(glob_avg, axis=1)
        glob_avg = tf.expand_dims(glob_avg, axis=1)

        #Batch Normalization
        bn = self.bn(glob_avg)
        
        #Activation
        activ = self.activation(bn)

        #Convolution operaion
        conv = self.conv1(activ)
        
        #Upsampling to match the size
        upsamp = self.upsample(conv)
        
        return  upsamp  
 
class context_flow(tf.keras.layers.Layer): 
    "Building COntext flow block"
    def __init__(self, name="context_flow"):
        super().__init__(name=name)

        #All necessary blocks
        self.concate = Concatenate()

        self.avg_pool = AveragePooling2D(pool_size=(2,2))

        self.conv_fusion1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')
        self.conv_fusion2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')
        self.conv_refin1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')
        self.conv_refin2 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')

        self.activ_relu = Activation("relu")
        self.activ_sigmoid = Activation("sigmoid")
        
        self.add = Add()
        self.multiply = Multiply()
        
        self.upsample = UpSampling2D(size=(2,2),interpolation='bilinear') 
        
    def call(self, X):
        # here X will a list of two elements, one from conv_block4 and  another from the flow of model
        inp, flow = X[0], X[1] 
        
        # Context Fusion Module

        #Concactination
        concat = self.concate([inp, flow])
        #Global avg pooling
        avg_pool = self.avg_pool(concat)
        #Convolution
        convf1 = self.conv_fusion1(avg_pool)
        convf2 = self.conv_fusion2(convf1)

        # Context Refinment Module

        #Convolution
        convr1 = self.conv_refin1(convf2)
        #Relu activation
        activ_relu = self.activ_relu(convr1)
        #Convoluition
        convr2 = self.conv_refin2(activ_relu)
        #Sigmoid activation
        activ_sig = self.activ_sigmoid(convr2)
        #Multiply
        mult = self.multiply([convf2, activ_sig])
        #Addition
        add = self.add([convf2, mult])

        #Upsampling
        upsample = self.upsample(add)

        return upsample

###################################################################################################################
#                     Features Selection and Adapted Global Convolution Network Modules                           #
###################################################################################################################
class feature_selection_module(tf.keras.layers.Layer):  
    "Building Feature Selection Module"
    def __init__(self, name="feature_selection"):
        super().__init__(name=name)
        
        #All necessary blocks
        self.convin = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')
        self.convpar = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same')
        self.glob_avg = GlobalAveragePooling2D()
        self.bn = BatchNormalization(axis=3)
        self.activ_sig = Activation("sigmoid")
        self.mul = Multiply()

    def call(self, X):
        #Feature Selection Module
        
        #Convolution
        conv = self.convin(X)
        
        #Global Average Pooling
        glob_avg_pool = self.glob_avg(conv)
        glob_avg_pool = tf.expand_dims(glob_avg_pool, 1)
        glob_avg_pool = tf.expand_dims(glob_avg_pool, 1)

        #Convolution
        conv_par = self.convpar(glob_avg_pool)

        #Batch Normalization
        bn = self.bn(conv_par)
        
        #Activation
        activ = self.activ_sig(bn)

        #Multiply
        fsm_out = self.mul([conv, activ])

        return fsm_out
    
class adapted_global_conv_layer(tf.keras.layers.Layer):    
    "Buildind Adapted Global Covolutional Layer"
    
    def __init__(self, name="global_conv_net"):
        super().__init__(name=name)
        
        #All necessary blocks
        self.conv_left1 = Conv2D(32,kernel_size=(7,1),padding='same')
        self.conv_left2 = Conv2D(32,kernel_size=(1,7),padding='same')
        self.convr1 = Conv2D(32,kernel_size=(1,7),padding='same')
        self.convr2 = Conv2D(32,kernel_size=(7,1),padding='same')
        self.convfi = Conv2D(32,kernel_size=(3,3),padding='same')
        self.add = Add()
        
    def call(self, X):
        # Adapted Global Convolutional Network
        
        #Left path
        convL = self.conv_left1(X)
        convL = self.conv_left2(convL)

        #Right path
        convR = self.convr1(X)
        convR = self.convr2(convR)

        #Combining
        add = self.add([convL, convR])
        conv = self.convfi(add)
        out = self.add([add, conv])
        
        return out
    
###################################################################################################################
#                                        Building the CANet Network                                               #
###################################################################################################################
input = Input(shape=(128,128,3))

# Stage 1(Convolutional block C0)
X = Conv2D(64, (3, 3), name='conv1', padding="same", kernel_initializer=glorot_uniform(seed=0))(input)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)

###################################################################################################################
#                             Convolutional & Identity block from 1 to 4 layers                                   #
###################################################################################################################
# First Convolutional Block
c1 = convolutional_block(kernel=3,  filters=[4,4,8], stride=2, name='conv_block_C1')(X)
I11 = identity_block(kernel=3,  filters=[4,4,8], name='Identity_block_I11')(c1)

# Second convolutional block
c2 = convolutional_block(kernel=3, filters=[8,8,16], stride=2, name='conv_block_C2')(I11)
I21 = identity_block(kernel=3,  filters=[8,8,16], name='Identity_block_I21')(c2)
I22 = identity_block(kernel=3,  filters=[8,8,16], name='Identity_block_I22')(I21)

# Third convolutional block
c3 = convolutional_block(kernel=3, filters=[16,16,32], stride=1, name='conv_block_C3')(I22)
I31 = identity_block(kernel=3,  filters=[16,16,32], name='Identity_block_I31')(c3)
I32 = identity_block(kernel=3,  filters=[16,16,32], name='Identity_block_I32')(I31)
I33 = identity_block(kernel=3,  filters=[16,16,32], name='Identity_block_I33')(I32)

# Fourth convolutional block
c4 = convolutional_block(kernel=3, filters=[32,64,64], stride=1, name='conv_block_C4')(I33)
I41 = identity_block(kernel=3,  filters=[32,64,64], name='Identity_block_I41')(c4)
I42 = identity_block(kernel=3,  filters=[32,64,64], name='Identity_block_I42')(I41)
I43 = identity_block(kernel=3,  filters=[32,64,64], name='Identity_block_I43')(I42)
I44 = identity_block(kernel=3,  filters=[32,64,64], name='Identity_block_I44')(I43)

###################################################################################################################
#                              Chained Context Aggregation Model (CAM)                                            #
###################################################################################################################
# Global Flow
width = I44.shape[1]
height = I44.shape[2]
channel = I44.shape[-1]
GF = global_flow(width=width, height=height, channels=channel, name='Global_Flow')(I44)

# Context Flow 1
cf_in1 = [I44, GF]
CF1 = context_flow(name='Context_flow_CF1')(cf_in1)

# Context Flow 2
cf_in2 = [I44, CF1]
CF2 = context_flow(name='Context_flow_CF2')(cf_in2)

# Context Flow 3
cf_in3 = [I44, CF2]
CF3 = context_flow(name='Context_flow_CF3')(cf_in3)

# Summation of Global and Context flows
flow_sum = Add()([GF, CF1, CF2, CF3])

# Feature Selection Module (FSM)
fsm = feature_selection_module(name='Feature_Selection_Module')(flow_sum)

#Upsamplng FSM output
fsm_upsample = UpSampling2D(size=(2,2), interpolation='bilinear')(fsm)

###################################################################################################################
#                                                     Decoder                                                     #
###################################################################################################################
# Adaptive Gloabl Convolutional Layer
agcn = adapted_global_conv_layer(name='Adaptive_Global_Conv_Layer')(c1)

# Concatenation of FSM and AGCN
concat = Concatenate()([fsm_upsample, agcn])

# Final Convolution Layer
final_conv = Conv2D(filters=21, kernel_size=(3,3), strides=(1,1), padding='same', name='final_conv')(concat)

# Upsampling for prediction
final_upsample = UpSampling2D(size=(4,4), interpolation='bilinear')(final_conv)

# Activation
output = Activation("softmax")(final_upsample)

#model
model = Model(inputs = input, outputs = output)
