import os
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
import models
import tensorflow as tf

from utils import load_imgs
import random

nb_epoch=200
height,width =512,512
model_name = 'lite_kits'
model_filename = "saved_models/{}.h5".format(model_name)
if not os.path.exists('saved_models'): os.mkdir('saved_models')

def weighted_binary_crossentropy(y_true, y_pred,weight = 0.666/0.333):
    eps = K.epsilon()  #10e-6
    y_pred = K.clip(y_pred,eps,1.-eps)
    return K.mean(-weight*y_true*K.log(y_pred)-(1.-y_true)*K.log(1.-y_pred))

def focal_loss_fixed(y_true, y_pred):
    # tensorflow backend, alpha and gamma are hyper-parameters which can set by you
    alpha = 0.66
    gamma = 2 # 0.75
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


model = models.lite_conv(height,width, loss=weighted_binary_crossentropy, optimizer = Adam(lr=1e-4), metrics = ['accuracy'],channels=1)

print ("Loading images")
imgs, masks ,labels= load_imgs(random.sample(range(0,140),20)) #随机产生不重复整数列表

if os.path.exists("saved_models/{}_1.h5".format(model_name)):
    print('loading model')
    model = load_model("saved_models/{}_1.h5".format(model_name),custom_objects={'weighted_binary_crossentropy':weighted_binary_crossentropy})

print ('Fit model')
model_checkpoint = ModelCheckpoint(model_filename, monitor= 'val_acc', save_best_only=True, verbose=1)  
model.summary()


history = model.fit(imgs,labels,batch_size = 4,shuffle=True,epochs=nb_epoch, 
  callbacks=[model_checkpoint],verbose=1,validation_split = 0.4)
model.save(model_filename)

imgs, masks ,labels= load_imgs(random.sample(range(0,140),6)) #随机产生不重复整数列表
loss = model.evaluate(imgs,labels,batch_size=10,verbose=0)
print(loss)