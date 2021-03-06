import tensorflow as tf
from model.Blocks import RDB,SDAB,MAM

'''Model Design

Achieive pursuit problem : Y=X1+X2+noise --> observe Y , split Y into X1 and X2  

Main Property : the inherent correlation among the rain steaks within an image
should be observably stronger than that between the rain streaks and the background

First Stage : rain streak component (denoted by R1) is initially generated

Second Stage : The main purpose of integrating the initial rain component
(R1) and the input rainy image (O) is to assist in the second
stage of locating possible rain pixels while suppressing
non-rain pixels in the image

Why element-wise addition operation (Conv + ReLU) ? : The main advantage
of this element-wise addition operation is to enhance the
shallow layer features for the second stage of the proposed
deep learning model to learn the distribution of rain streaks.
'''



class SSDRNet(tf.keras.Model):
    def __init__(self):
        super(SSDRNet,self).__init__()

        self.conv_first_stage=tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
        )


        self.conv_second_stage=tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
        )

        self.conv_output = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None
        )

        self.feature_extractor1=tf.keras.Sequential([RDB() for _ in range(3)])
        self.refinement=tf.keras.Sequential([SDAB() for _ in range(4)])
        self.feature_extractor2=tf.keras.Sequential([RDB() for _ in range(2)])
        self.transformer=MAM()

        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self,x,training=False):
        '''
        x:(batch,h,w,3)
        '''
        #first stage
        x1=self.conv_first_stage(x)
        F1=self.feature_extractor1(x1,training=training)
        F1=self.refinement(F1,training=training)
        F1=self.feature_extractor2(F1,training=training)
        R1=self.transformer(F1,training=training)


        #second stage
        H=self.concat([R1,x])
        x2=self.conv_second_stage(H)
        F2=tf.add(x1,x2)
        F2=self.feature_extractor1(F2,training=training)
        F2=self.refinement(F2,training=training)
        F2=self.feature_extractor2(F2,training=training)
        R2=self.transformer(F2,training=training) #R2=C1 or C2


        #X=Component1+Component2+noise , get desired C=C1 or C2
        C_with_noise=tf.subtract(x,R2) #C+noise=X-R2
        denoise=self.conv_output(C_with_noise) #denoise=-noise=conv(C+noise)
        C=tf.add(C_with_noise,denoise) #C+noise+denoise=C=desired Component
        return R1,R2,C


#a=tf.random.normal((1,67,67,3))
#m=SSDRNet()
#m.load_weights('aaa')
#x=m(a)
#m.save_weights('aaa')





'''
input:normalized image
??????output :  ???output restrict ??? 0~1 and then *255???get???image???
'''




