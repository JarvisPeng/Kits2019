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
import glob
from sklearn.model_selection import train_test_split
import cv2 
import tensorflow as tf
import random
from utils import load_imgs

# For multi_gpu use uncommemt and set visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def muti_jacc_coef(y_true, y_pred, smooth = 0.001,num_class=2):
    smooth = K.epsilon()
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true,axis=[1,2]) + K.sum(y_pred,axis=[1,2])- intersection
    loss =  (intersection + smooth) /  (union + smooth)
    return K.mean(loss)

def muti_dice_coef(y_true, y_pred, smooth=0.001,num_class=2):
    smooth = K.epsilon()
    intersection = K.sum(y_true * y_pred, axis=[1,2])       #在图片的宽高轴求和得到交集 ，最后直接求总平均即可得到dice平均
    sum_area = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    loss = (2. * intersection + smooth) / (sum_area + smooth)
    return  K.mean(loss)  # K.mean(mean_loss,axis=0)

def muti_dice_coef_loss(y_true, y_pred):
    return 1 - muti_dice_coef(y_true, y_pred)


height, width = 512, 512
nb_epoch = 200
batch_size = 1
model_name = "kits2019"
model_filename = "saved_models/{}.h5".format(model_name)
print(model_filename)
if not os.path.exists('saved_models'): os.mkdir('saved_models')

model = models.get_unet(height, width, loss=[muti_dice_coef_loss], optimizer = Adam(lr=1e-4), 
              metrics = [muti_jacc_coef,'accuracy',sensitivity,specificity,muti_dice_coef],channels=1, num_class=1)

if os.path.exists("saved_models/{}_1.h5".format(model_name)):
    print('loading model')
    model = load_model("saved_models/{}_1.h5".format(model_name),
        custom_objects={'muti_dice_coef_loss':muti_dice_coef_loss,'muti_jacc_coef':muti_jacc_coef,'muti_dice_coef':muti_dice_coef,
                        'sensitivity':sensitivity,'specificity':specificity})

model.summary()

# ##########训练training
# print ("Loading images")
# imgs, masks ,labels= load_imgs(random.sample(range(0,140), 1),no_negative=True) #随机产生不重复整数列表
# print ('Fit model')
# model_checkpoint = ModelCheckpoint(model_filename, monitor= 'val_muti_jacc_coef', save_best_only=True, verbose=1)  # 'val_s_jacc_coef'
# history = model.fit(imgs,masks,batch_size = 1,shuffle=True,epochs=nb_epoch, 
#   callbacks=[model_checkpoint],verbose=1,validation_split = 0.4)

# ########评估测试
def evaluate():
    num, loss_all = 0, 0
    for i in range(140,209):
        case = [i]
        no_negative ,kidney_only = True  ,True
        imgs_test_p, masks_test ,labels_test=load_imgs(case,no_negative=no_negative,kidney_only=kidney_only,normalization=True)
        imgs_test, masks_test ,labels_test=load_imgs(case,no_negative=no_negative,kidney_only=kidney_only,normalization=False)
        loss = model.evaluate(imgs_test_p,masks_test,batch_size=1,verbose=0)
        loss_all=(num*loss_all+loss[1]*labels_test.shape[0])/(num+labels_test.shape[0])
        num +=labels_test.shape[0]
        print(loss,num,loss_all)

###########查看分割结果     
#l = list(range(15,25))  
l = (random.sample(range(0,50),10))
case = [46]
no_negative ,kidney_only = False  ,False
imgs_test_p, masks_test ,labels_test=load_imgs(case,no_negative=no_negative,kidney_only=kidney_only,normalization=True)
imgs_test, masks_test ,labels_test=load_imgs(case,no_negative=no_negative,kidney_only=kidney_only,normalization=False)
masks_p = model.predict(imgs_test_p[l],batch_size=1,verbose=1)
masks_p = np.round(np.squeeze(masks_p[:,:,:,0])).astype(np.uint8)       # 0:kidney  1:tumot
masks_t_k = np.squeeze(masks_test[:,:,:,0]).astype(np.uint8)     # mask of true kidney 
masks_t_t = np.squeeze(masks_test[:,:,:,1]).astype(np.uint8)     # mask of true tumor
imgs_test = np.squeeze(imgs_test).astype(np.uint8)
for index,i in enumerate(l):
    
    #  ret, img_binary = cv2.threshold(masks_t[i].astype(np.uint8),0.5,255,cv2.THRESH_BINARY)
    # contour 是一个包含所有轮廓的list 输入图像位二值图像np.uint8 
    _,contour_t_k,_=cv2.findContours(masks_t_k[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _,contour_t_t,_=cv2.findContours(masks_t_t[i],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _,contour_p,_=cv2.findContours(masks_p[index],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 灰度转RGB图像，复制三个矩阵用于作轮廓线  truth ,predict, truth_and_predict
    img_o = cv2.cvtColor(imgs_test[i], cv2.COLOR_GRAY2RGB)  
    img_t, img_p, img_t_p = np.copy(img_o), np.copy(img_o),np.copy(img_o)
    
    imgs_test[i] = np.where(masks_t_k[i],imgs_test[i],0)

    # 原始图像，轮廓列表，-1表示绘制所有轮廓线，颜色，线宽(int)
    thickness = 2 
    cv2.drawContours(img_t,contour_t_k,-1,(255,0,0),thickness) # 参数
    cv2.drawContours(img_p,contour_p,-1,(0,0,255),thickness) 

    # true and prdict 轮廓
    cv2.drawContours(img_t_p,contour_t_k,-1,(255,0,0),thickness) 
    cv2.drawContours(img_t_p,contour_t_t,-1,(0,128,0),thickness)
    cv2.drawContours(img_t_p,contour_p,-1,(0,0,255),thickness) 

    # ######分别查看
    # f,ax = plt.subplots(2,3,figsize=(12,9))
    # ax[0,0].imshow(imgs_test[i],'gray')
    # ax[0,1].imshow(masks_t_k[i],'gray')
    # ax[0,2].imshow(masks_p[index],'gray')
    # ax[1,0].imshow(img_t_p)
    # ax[1,1].imshow(img_t)
    # ax[1,2].imshow(img_p)
    # plt.show()

    plt.figure(figsize=(12,9))
    plt.imshow(img_t_p)
    plt.show()
