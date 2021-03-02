#!/usr/bin/env python
# coding: utf-8

# #  <center> TrafficSignNet
# Conv2D   
# input: (None, 32, 32, 3)   
# output: (None, 32, 32, 8) 
#     
# Activation   
# input: (None, 32,32,8)   
# output: (None, 32,32,8) 
#     
# BatchNormalization   
# input: (None, 32, 32, 8)   
# output: (None, 32, 32, 8) 
#     
# MaxPooling2D   
# input: (None, 32, 32, 8)   
# output: (None, 16, 16, 8) 
# 

# In[2]:


#necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dropout


# In[3]:


class TrafficSignNet:
    
    def build(width, height, channel, classes):
        
        #init model
        model = Sequential()
        inputShape = (height, width, channel)
        chanDim = -1
        
        # CONV => RELU => BN => POOL
        
        model.add(Conv2D(8, (5, 5), padding="same", input_shape= inputShape))       
        # 5×5 kernel to learn larger features
        # distinguish between different traffic sign shapes and color blobs
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
                  
        #(CONV => RELU => CONV => RELU) * 2 => POOL layers:
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #The head of our network consists of two sets of fully connected layers and a softmax classifier
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # second set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
                  
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
## if you cant get this model goto https://github.com/yeasin50/startUP_CNN
        


# In[ ]:




