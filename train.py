from model.networks import SSDRNet
from model.loss_function import hybrid_loss
import tensorflow as tf
from tqdm import tqdm
from preprocessing import preprocess_train,preprocess_test,augmentation
import numpy as np

'''labels
clear : gt_C
rain streak : gt_R
'''

train_path='./data/train/'
test_path='./data/test/'
epochs=100
train_batch=2 #my compute gpu sucks
test_batch=32
lr=0.001

normalizer=tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)

def train(x,gt_C,gt_R,model,optimizer):
    with tf.GradientTape() as tape:
        R1,R2,C = model(x,training=True)
        loss=hybrid_loss(R1,gt_R)+hybrid_loss(R2,gt_R)+hybrid_loss(C,gt_C)
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss.numpy()


def evaluate_test(model,ds_test):
    loop = tqdm(ds_test, leave=True)
    total_ssim=[]
    for x,gt_C in loop:
        x,gt_C = normalizer(x[0][:,0,...]),normalizer(gt_C[0][:,0,...])
        R1, R2, C = model(x, training=False)
        ssim_score=tf.reduce_mean(tf.image.ssim(C,gt_C,1.0))
        total_ssim.append(ssim_score.numpy())
        loop.set_postfix(loss=ssim_score.numpy())
    return np.mean(total_ssim)


def train_loop():
    ds_train=preprocess_train(train_path,train_batch)
    ds_test=preprocess_test(test_path,test_batch)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=4,
        decay_rate=0.9)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model=SSDRNet()

    prev_test_ssim_score=0.
    for epoch in range(epochs):
        loop = tqdm(ds_train, leave=True)
        for x,gt_C,gt_R in loop:
            #x,gt_C,gt_R=augmentation(x[0][:,0,...],gt_C[0][:,0,...],gt_R[0][:,0,...])
            x, gt_C, gt_R = x[0][:, 0, ...], gt_C[0][:, 0, ...], gt_R[0][:, 0, ...]
            x,gt_C,gt_R=normalizer(x),normalizer(gt_C),normalizer(gt_R)
            loss=train(x,gt_C,gt_R,model,optimizer)
            loop.set_postfix(loss=f'loss:{loss}')

        test_ssim_score=evaluate_test(model,ds_test)
        loop.set_postfix(loss=f'test_ssim_score:{test_ssim_score}')

        if test_ssim_score>0.7 and test_ssim_score>prev_test_ssim_score:
            print('save weighs')
            model.save_weights('./save/ssdr_weights')
            prev_test_ssim_score=test_ssim_score

if __name__ == '__main__':
    train_loop()
        
