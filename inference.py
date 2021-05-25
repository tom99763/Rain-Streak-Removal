import tensorflow as tf
from model.networks import SSDRNet
import cv2
import numpy as np


model=SSDRNet()
model.load_weights('./save/ssdr_weights')
norm=tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)

img=cv2.imread('./data/test/img/rainy_images/rain-089.png')
img=img[np.newaxis,...].astype('float32')
img=norm(img)
R1,R2,C=model(img,training=False)


R1=tf.squeeze(tf.clip_by_value(R1,0,1)*255,axis=0).numpy().astype('uint8')
R2=tf.squeeze(tf.clip_by_value(R2,0,1)*255,axis=0).numpy().astype('uint8')
C=tf.squeeze(tf.clip_by_value(C,0,1)*255,axis=0).numpy().astype('uint8')

cv2.imshow('r3',C)
cv2.imshow('r2',R1)
cv2.imshow('r1',R2)
cv2.waitKey(0)