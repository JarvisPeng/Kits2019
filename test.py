from starter_code.utils import load_case
import numpy as np
# volume, segmentation = load_case("case_00123")
# or
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

def togray(volume, hu_min, hu_max):
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

def seg_to_mask(seg,class_nums = 2):

	positve = np.ones(seg.shape).astype(np.uint8)
	negative = np.zeros(seg.shape).astype(np.uint8)
	# 两类
	kidney = np.where(seg==1,positve,negative)
	tumor = np.where(seg==2,positve,negative)
	mask = np.stack((kidney,tumor),axis=-1)
	# print(mask.shape)
	return mask

def load_imgs(nums=210,height=512,width=512,channels=1, num_class=2):
	# imgs =np.zeros(shape=(nums,height,width,channels))
	# masks =np.zeros(shape=(nums,height,width,channels))
	for i in range(0, nums):
		volume, segmentation = load_case(i)
		img = volume.get_data()
		# print(img.shape)
		img = togray(img,-512,512).astype(np.uint8)
		seg = segmentation.get_data().astype(np.uint8)
		img = np.expand_dims(img,axis=-1)
		# print(img.shape)
		mask = seg_to_mask(seg)
		label = np.array([np.max(mask[i]) for i in range(mask.shape[0])]).astype(np.uint8)  #判断是否有分割内容
		if i ==0: imgs,masks,labels= img,mask,label
		else:
			imgs = np.concatenate([imgs,img],axis=0)
			masks = np.concatenate([masks,mask],axis=0)
			labels = np.concatenate([labels,label],axis=0)
		# img=[img[i] for i in range(img.shape[0])]
		# mask=[mask[i] for i in range(mask.shape[0])]
		# imgs+=img
		# masks+=mask
		# np.save('imgs/imgs{}.npy'.format(i), imgs) 
		# np.save('segs/segs{}.npy'.format(i), segs) 
		# print(i,imgs.shape,masks.shape)
		if i%10==0: print(i,imgs.shape,masks.shape,np.mean(labels))

	return imgs, masks, labels
load_imgs(2)

# volume, segmentation = load_case(1)
# img = volume.get_data()
# seg = segmentation.get_data()
# OrthoSlicer3D(img).show()
# img1=togray(img[46],-512,512).astype(np.uint8)
# seg1=seg[46].astype(np.uint8)
# # figutre, ax = plt.subplots(1,2)
# plt.subplot(1,2,1)
# plt.imshow(img1,'gray')
# plt.subplot(1,2,2)
# plt.imshow(seg1,'gray')
# plt.show()

# print(np.amax(img),np.amin(img), img.shape,np.average(img),np.std(img),seg.shape,np.mean(seg))
# print(volume.shape,type(volume),np.array(volume),np.array(segmentation))


# from starter_code.visualize import visualize

# visualize("case_00123", <destination (str)>)
# or
# visualize(6,'test6_1/')
# 

