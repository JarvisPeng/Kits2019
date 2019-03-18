import os
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean,dice_coef,dice_jacc_single,sensitivity,specificity
import models
import glob
from sklearn.model_selection import train_test_split
import cv2 

np.random.seed(3)
K.set_image_dim_ordering("tf")  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow

# For multi_gpu use uncommemt and set visible devices

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Abstract: https://arxiv.org/abs/1703.04819

# Extract challenge Training / Validation / Test images as below
# Download from https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a

training_folder = "../Test_Data/Images"     #测试可运行用1
training_mask_folder = "../Test_Data/Segmentation"



def img_sensitivity(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.bool)
    y_pred = y_pred.reshape(-1).astype(np.bool)

    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + np.finfo(float).eps)

def img_specificity(y_true, y_pred):

    y_true = y_true.reshape(-1).astype(np.bool)
    y_pred = y_pred.reshape(-1).astype(np.bool)
    true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))

    possible_negatives = np.sum(np.round(np.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + np.finfo(float).eps)
def img_accuracy(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.bool)
    y_pred = y_pred.reshape(-1).astype(np.bool)

    true_negatives = np.sum(np.round(np.clip((1-y_true) * (1-y_pred), 0, 1)))
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    denom = y_true.shape[0]
    return (true_negatives+true_positives) / (denom + np.finfo(float).eps)

def post_process(input_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opening = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ret,thresh = cv2.threshold(closing,127,255,0)
    im2,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max_area = 0
    max_index = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area>cnt_max_area:
            cnt_max_area = area
            max_index = i
    img = np.zeros(input_mask.shape,dtype=np.uint8)
    if len(contours)>=1:
        output_mask = cv2.drawContours(img, [contours[max_index]], 0,(255,255,255), -1)
    else:
        output_mask = img
    return output_mask


# y_true and y_pred should be one-hot
# y_true.shape = (None,Width,Height,Channel)
# y_pred.shape = (None,Width,Height,Channel)
def muti_dice_coef(y_true, y_pred, smooth=1,num_class=2):
    mean_loss = 0;
    for i in range(num_class):  #y_pred.shape(-1)):
        y_true_f = K.flatten(y_true[:,:,:,i],axis=-1)
        y_pred_f = K.flatten(y_pred[:,:,:,i],axis=-1)
        intersection = K.sum(y_true_f*y_pred_f)
        union = K.sum(y_true_f)+K.sum(y_pred_f)
        # intersection = K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=[1,2,3])
        # union = K.sum(y_true[:,:,:,i], axis=[1,2,3]) + K.sum(y_pred[:,:,:,i], axis=[1,2,3])
    mean_loss += (2. * intersection + smooth) / (union + smooth)
    return K.mean(mean_loss, axis=0)
def muti_dice_coef_loss(y_true, y_pred):
    1 - dice_coef(y_true, y_pred)



seed = 1
height, width = 128, 128
nb_epoch = 200
model_name = "final_2018_3"

do_train = True # train network and save as model_name
do_predict = False # use model to predict and save generated masks for Validation/Test
do_ensemble = False # use previously saved predicted masks from multiple models to generate final masks
ensemble_pkl_filenames = ["model1", "model3","model5"]
model = 'unet'
batch_size = 4
loss_param = 'dice'
optimizer_param = 'adam'
monitor_metric = 'val_jacc_coef'


metrics = [jacc_coef,'accuracy',sensitivity,specificity,dice_coef]


    
loss_options = {'BCE': 'binary_crossentropy', 'dice':dice_loss, 'jacc':jacc_loss, 'mse':'mean_squared_error'}
optimizer_options = {'adam': Adam(lr=1e-4),
                     'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)}

loss = loss_options[loss_param]
optimizer = optimizer_options[optimizer_param]
model_filename = "saved_models/2019{}.h5".format(model_name)
print(model_filename)

model = models.get_unet(512,512, loss=[loss,'binary_crossentropy'], optimizer = optimizer, metrics = metrics,channels=1, num_class=2)
# model = models.get_unet(height,width, loss=muti_dice_coef_loss, optimizer = optimizer, metrics = muti_dice_coef,channels=1, num_class=2)


print ("Loading images")


from test import load_imgs
imgs, masks ,labels= load_imgs(1)

print ("Using batch size = {}".format(batch_size))
print ('Fit model')
model_checkpoint = ModelCheckpoint(model_filename, monitor= 'val_jacc_coef', save_best_only=True, verbose=1)

model.summary()
print ("Not using validation during training")
history = model.fit(imgs,[masks,labels],
    batch_size = 1,
   epochs=nb_epoch, 
  callbacks=[model_checkpoint],verbose=1,
  validation_split = 0.2)
model.save(model_filename)