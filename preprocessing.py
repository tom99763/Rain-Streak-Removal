import tensorflow as tf
import numpy as np

def augmentation(img,gt_C,gt_R):
    if np.random.binomial(1,0.5):
        img = tf.image.flip_left_right(img)
        gt_R= tf.image.flip_left_right(gt_R)
        gt_C= tf.image.flip_left_right(gt_C)
    return img,gt_C,gt_R


def preprocess_train(data_path,batch_size):

    ds_img = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path+'/img',
        shuffle=False,
        image_size=(128,128),
        batch_size=1
    )


    ds_gt_C = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path+'/gt_C',
        shuffle=False,
        image_size=(128,128),
        batch_size=1
    )

    ds_gt_R = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path+'/gt_R',
        shuffle=False,
        image_size=(128,128),
        batch_size=1
    )

    ds =tf.data.Dataset.zip((ds_img,ds_gt_C,ds_gt_R)).batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def preprocess_test(data_path,batch_size):
    ds_img = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path+'/img',
        shuffle=False,
        image_size=(128,128),
        batch_size=1
    )


    ds_gt_C = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data_path+'/gt_C',
        shuffle=False,
        image_size=(128,128),
        batch_size=1
    )

    ds =tf.data.Dataset.zip((ds_img,ds_gt_C)).batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

'''
data_path ='./data/test/'

for x1,x2 in preprocess_test(data_path,32):
    print(x1[0].shape,x2[0].shape)
    break
'''

'''
data_path='./data/train/'
for x1,x2,x3 in preprocess_train(data_path,32):
    print(x1[0].shape,x2[0].shape,x3[0].shape)
    break
'''
