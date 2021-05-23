import tensorflow as tf

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs

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

    ds =tf.data.Dataset.zip((ds_img,ds_gt_C,ds_gt_R)).batch(batch_size)
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

    ds =tf.data.Dataset.zip((ds_img,ds_gt_C)).batch(batch_size)
    return ds
