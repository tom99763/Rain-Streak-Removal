import tensorflow as tf


'''labels、loss
gt:clear_img 、 pure_rain_streak
pred:R1、R2、desired_clear_img
loss=loss_hybrid_R1+loss_hybrid_R2+loss_hybrid_desired
'''

l1 = tf.keras.losses.MeanAbsoluteError()
def hybrid_loss(gen,gt):
    l1_loss=l1(gt,gen) #l1 distance of two image
    ssim_loss= 1 - tf.reduce_mean(tf.image.ssim(gt, gen, max_val=1.0)) #structure similarity of two image, sum over batch
    loss=l1_loss+ssim_loss
    return loss
