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
epochs=50
train_batch=18 #my compute gpu sucks
test_batch=100
lr=0.001
load_weights=False

normalizer=tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)

def train(x,gt_C,gt_R,model,optimizer):
    with tf.GradientTape() as tape:
        R1,R2,C = model(x,training=True)
        loss=hybrid_loss(R1,gt_R)+hybrid_loss(R2,gt_R)+hybrid_loss(C,gt_C)
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss


def evaluate_test(model,ds_test):
    total_ssim=[]
    print('test time')
    for x,gt_C in ds_test:
        x,gt_C = normalizer(x[0][:,0,...]),normalizer(gt_C[0][:,0,...])
        R1, R2, C = model(x, training=False)
        C = tf.clip_by_value(C, 0, 1)
        ssim_score=tf.reduce_mean(tf.image.ssim(C,gt_C,1.0))
        total_ssim.append(ssim_score.numpy())
    return np.mean(total_ssim)


def train_loop():
    ds_train=preprocess_train(train_path,train_batch)
    ds_test=preprocess_test(test_path,test_batch)
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    model=SSDRNet()
    if load_weights:
        print('load weights')
        model.load_weights('./save/ssdr_weights')

    prev_test_ssim_score=0.
    for epoch in range(epochs):
        loop = tqdm(ds_train, leave=True)
        for i,(x,gt_C,gt_R) in enumerate(loop):
            #x,gt_C,gt_R=augmentation(x[0][:,0,...],gt_C[0][:,0,...],gt_R[0][:,0,...])
            x, gt_C, gt_R = x[0][:, 0, ...], gt_C[0][:, 0, ...], gt_R[0][:, 0, ...]
            x,gt_C,gt_R=normalizer(x),normalizer(gt_C),normalizer(gt_R)
            loss=train(x,gt_C,gt_R,model,optimizer)
            loop.set_postfix(loss=f'epoch:{epoch},no.{i},loss:{loss}')

        test_ssim_score=evaluate_test(model,ds_test)
        print(f'test_ssim_score:{test_ssim_score}')

        if test_ssim_score>0.85 and test_ssim_score>prev_test_ssim_score:
            print('save weighs')
            model.save_weights('./save/ssdr_weights')
            prev_test_ssim_score=test_ssim_score

if __name__ == '__main__':
    train_loop()
        
