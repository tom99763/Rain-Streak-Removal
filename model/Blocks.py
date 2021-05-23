import tensorflow as tf

#Residual dense blocks
#proposed from this paper
#Roll : Feature Extraction
class RDB(tf.keras.layers.Layer):
    def __init__(self,in_channel =32):
        super(RDB,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(
            filters=in_channel,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv2=tf.keras.layers.Conv2D(
            filters=in_channel,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.concat=tf.keras.layers.Concatenate(axis=-1)

        self.activation1=tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2])
        self.activation2=tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2])

    def call(self,x,training=False):
        '''
        x:(batch,h,w,32)
        output : (batch,h,w,32)
        '''
        F_1=self.activation1(self.conv1(x))
        F_2=self.concat([F_1,x])  #32-->64
        F_3=self.conv2(F_2) #64-->32
        F_3=tf.add(F_3,x) #Residual
        output=self.activation2(F_3)
        return output
'''Test RDB Block
a=tf.random.normal((1,64,64,32))
m=RDB()
print(m(a).shape)
'''

#channel attention
#consider information in spatial domain to give attention to channel domain
class Component_Attention(tf.keras.layers.Layer):
    def __init__(self,in_channel = 32):
        super(Component_Attention,self).__init__()
        self.conv= tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.activation = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(units=in_channel//2,activation=None),
            tf.keras.layers.PReLU(tf.constant_initializer(0.25)), #no shared axis
            tf.keras.layers.Dense(units=in_channel,activation=None)
        ])

    def call(self,x,training=False):
        '''
        x:(batch,h,w,32) , roll:value、key
        '''
        batch, h, w, c = x.shape

        #build attention
        B1 = tf.reshape(x, shape=(batch, h * w, c))  # batch,hw,32 , roll:key
        M1 = self.activation(self.conv(x))  # batch,h,w,1 , roll:non-softmax query*key
        B2 = tf.nn.softmax(tf.reshape(M1, shape=(batch, h * w, 1)), axis=1)  # batch,hw,1  roll:query*key

        # compute dot_similarity(B1,B2) in spatial domain to get attention of each channel---> 32
        M2 = tf.squeeze(tf.transpose(B1, perm=[0, 2, 1]) @ B2,axis=-1)  # batch,32  attention :  (query dot key) dot value
        CMap=tf.reshape(self.mlp(M2),shape=(batch,1,1,c)) #batch,1,1,32

        #broadcast add attention to channel domain of feature x
        COut=tf.add(x,CMap)  # batch,h,w,32 + batch,1,1,32 --> batch,h,w,32 ,
        return COut
'''Test Component Attention Block
a=tf.random.normal((1,64,64,32))
m=Component_Attention()
print(m(a).shape)
'''


#spatial attention
#consider information in channel domain to give attention to spatial domain
class Subsidiary_Attention(tf.keras.layers.Layer):
    def __init__(self,in_channel=32):
        super(Subsidiary_Attention,self).__init__()
        self.conv1= tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same'
        )

        self.conv2= tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same'
        )

        self.conv_dilated_1= tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=2
        )

        self.conv_dilated_2= tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=3
        )

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.activation1 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation_dilated_1 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation_dilated_2 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])


    def call(self,x,training=False):
        '''
        x:(batch,h,w,32) roll:key、value
        '''
        #3-head
        D1=self.activation1(self.conv1(x)) #batch,h,w,1
        D2=self.activation_dilated_1(self.conv_dilated_1(x)) #batch,h,w,1
        D3=self.activation_dilated_2(self.conv_dilated_2(x)) #batch,h,w,1

        #concat
        S1=self.concat([D1,D2,D3]) #batch,h,w,3

        S2=self.conv2(S1) #batch,h,w,1 roll:non-sigmoid query*key
        SMap=tf.nn.sigmoid(S2) #batch,h,w,1 , query*key

        #give spatial attention to spatial domain of feature x
        SOut=tf.multiply(SMap,x) # attention : (query dot key) dot value
        return SOut
'''Test Subsidiary_Attention
a=tf.random.normal((1,64,64,32))
m=Subsidiary_Attention()
print(m(a).shape)
'''



#Sequential Dual attention blocks
#https://arxiv.org/pdf/1809.02983.pdf
class SDAB(tf.keras.layers.Layer):
    def __init__(self,in_channel=32):
        super(SDAB,self).__init__()
        self.ca=Component_Attention()
        self.sa=Subsidiary_Attention()

        self.conv1= tf.keras.layers.Conv2D(
            filters=in_channel,
            kernel_size=3,
            strides=1,
            padding='same'
        )

        self.conv2= tf.keras.layers.Conv2D(
            filters=in_channel,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv3= tf.keras.layers.Conv2D(
            filters=in_channel,
            kernel_size=1,
            strides=1,
            padding='same'
        )

        self.activation1 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation2 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation3 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])

        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self,x,training=False):
        '''
        x:(batch,h,w,32)
        '''
        F_1=self.activation1(self.conv1(x)) #batch,h,w,32
        F_2=self.activation2(self.conv2(F_1)) #batch,h,w,32
        COut=self.ca(F_2,training=training) #batch,h,w,32
        SOut=self.sa(COut,training=training) #batch,h,w,32
        F_3=self.concat([COut,SOut]) #batch,h,w,64
        F_4=self.conv3(F_3) #batch,h,w,1
        F_4=tf.add(x,F_4) #residual
        out=self.activation3(F_4)
        return out
'''Test SDAB
a=tf.random.normal((1,64,64,32))
m=SDAB()
print(m(a).shape)
'''


#Multi-scale feature aggregation modules
#proposed from this paper
#transform feature representer to image domain
class MAM(tf.keras.layers.Layer):
    def __init__(self):
        super(MAM,self).__init__()
        self.conv1= tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv2= tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.conv_dilated_1= tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=2
        )
        self.conv_dilated_2= tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=3
        )
        self.conv_dilated_3= tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=4
        )

        self.activation1 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation_dilated_1 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation_dilated_2 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])
        self.activation_dilated_3 = tf.keras.layers.PReLU(tf.constant_initializer(0.25), shared_axes=[1, 2])

        self.concat = tf.keras.layers.Concatenate(axis=-1)
    def call(self,x,training=False):
        '''
        x:(batch,h,w,32)
        '''
        F1=self.activation1(self.conv1(x))
        F2=self.activation_dilated_1(self.conv_dilated_1(x))
        F3=self.activation_dilated_2(self.conv_dilated_2(x))
        F4=self.activation_dilated_3(self.conv_dilated_3(x))
        F=self.concat([x,F1,F2,F3,F4])
        out=self.conv2(F)
        return out #mixture feature
'''Test MAM
a=tf.random.normal((1,64,64,32))
m=MAM()
print(m(a).shape)
'''