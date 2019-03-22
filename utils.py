from starter_code.utils import load_case
import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

def to_gray(volume, hu_min, hu_max):
	volume = np.clip(volume, hu_min, hu_max)
	# 原始文件中灰度值范围为32位，把数值限制到255
	# Scale to values between 0 and 1
	mxval = np.max(volume)
	mnval = np.min(volume)
	im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

	# Return values scaled to 0-255 range, but *not cast to uint8*
	# Repeat three times to make compatible with color overlay
	im_volume = 255*im_volume
	return im_volume

def seg_to_mask(seg=[],kidney_only=True,class_nums = 2):

	positve = np.ones(seg.shape).astype(np.uint8)
	negative = np.zeros(seg.shape).astype(np.uint8)
	if kidney_only:
		mask = np.expand_dims(np.where(seg,positve,negative),axis=-1)
	else:
		# 两类   肾和肿瘤的分割可以重叠，肿瘤看成是肾的一部分
		kidney = np.where(seg,positve,negative)  # seg==1
		tumor = np.where(seg==2,positve,negative)
		mask = np.stack((kidney,tumor),axis=-1)
		print('shape of mask',mask.shape)
	
	return mask

def load_imgs(case_list=[0],height=512,width=512,channels=1, num_class=2,no_negative=False,normalization=False,kidney_only=True):
	for i in case_list:
		volume, segmentation = load_case(i)
		img = volume.get_data()
		# print(img.shape)
		img = to_gray(img,-512,512).astype(np.uint8)
		seg = segmentation.get_data().astype(np.uint8)
		if normalization: img = (img-np.mean(img))/np.std(img)	#是否归一化
		img = np.expand_dims(img,axis=-1)
		# print(img.shape)
		mask = seg_to_mask(seg,kidney_only)
		label = np.array([np.max(mask[i]) for i in range(mask.shape[0])]).astype(np.uint8)  #判断是否有分割内容
						 #  np.amax(mask[i],axis=(0,1))[1]		#肾和肿瘤都训练时，选出包含两种内容的图片
		if no_negative:		#不加载不含分割图的
			l = np.squeeze(np.argwhere(label==1)).tolist()  # positve_list
			img,mask,label = img[l],mask[l],label[l]

		if i == case_list[0]: imgs,masks,labels= img,mask,label
		else:
			imgs = np.concatenate([imgs,img],axis=0)
			masks = np.concatenate([masks,mask],axis=0)
			labels = np.concatenate([labels,label],axis=0)

		print(i,img.shape[0],np.mean(label))

	print('all images',imgs.shape[0],np.mean(labels))
	return imgs, masks, labels
# load_imgs(2)


# ###### 3视图查看分割结果和原始图片
# volume, segmentation = load_case(188)
# img = volume.get_data()
# seg = segmentation.get_data()
# print(volume)
# s = seg.shape  			#函数OrthoSlicer3D显示的时候是按比例来显示的，所以乘以2或1.5，并区分k和t两类，更清晰一点
# seg = (seg*2+(np.random.randn(s[0],s[1],s[2])))		#加入很小随机数，调用函数OrthoSlicer3D才能运行出有效可视结果	
# OrthoSlicer3D(img).show()



# ######
# from starter_code.visualize import visualize
# visualize("case_00123", <destination (str)>)
# or
# visualize(6,'test6_1/')
# 

