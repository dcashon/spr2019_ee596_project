import numpy as np
import tensorflow as tf

# D. Cashon
# 2019 06 02

class SimpleBDDModel():
    def __init__(self, activation=tf.nn.relu):
        """
        Constructs the model
        Input is assumed to be 256 by 256 by 3

        """
        
        self.X =  tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
        self.Y1 = tf.placeholder(tf.float32, shape=(None, 10))  # 10 class
        self.Y2 = tf.placeholder(tf.float32, shape=(None, 4)) # regression bbox
        #self.training = tf.placeholder(tf.bool)

        # FEATURE MAPPER
        with tf.variable_scope("feature_map"):
            # --------------------Layer 1---------------------
            c1 = tf.layers.conv2d(self.X, filters=96, kernel_size=7, strides=3,
               name='conv1')
            a1 = activation(c1, name='act1')
            #b1 = tf.layers.batch_normalization(a1, training=self.training, name='batch1');
            p1 = tf.layers.max_pooling2d(a1, pool_size=2, strides=2,
               name='maxpool1')
            # -------------------------------------------------
            # ------------------Layer 2------------------------
            c2 = tf.layers.conv2d(p1, filters=256, kernel_size=3, strides=1,
                    name='conv2')
            a2 = activation(c2, name='act2')
            # -------------------------------------------------
            # ------------------Layer 3------------------------
            c3 = tf.layers.conv2d(a2, filters=512, kernel_size=3, strides=1,
                    name='conv3')
            a3 = activation(c3, name='act3')
            p3 = tf.layers.max_pooling2d(a3, pool_size=2, strides=2,
            name='maxpool2')
            # -------------------------------------------------
            #b2 = tf.layers.batch_normalization(a2, training=self.training, name='batch2')

        # CLASSIFICATION HEAD
        # output of p4: 7 by 7 by 1024
        with tf.variable_scope("classifier"):
            f5 = tf.layers.conv2d(p3, filters=3072, kernel_size=19, strides=1,
               name='fconv1')
            f6 = tf.layers.conv2d(f5, filters=4096, kernel_size=1, strides=1,
               name='fconv2')
            self.logits = tf.layers.conv2d(f6, filters=10, kernel_size=1,strides=1,
               name='class_out')
            self.squeeze_logits = tf.squeeze(self.logits,
            name='class_out_squeeze')

        # REGRESSOR HEAD
        #with tf.variable_scope("regressor"):
        #    f7 = tf.layers.conv2d(p1, filters=4096, kernel_size=42, strides=1,
        #            name='fconv3')
        #    f8 = tf.layers.conv2d(f7, filters=1024, kernel_size=1, strides=1,
        #            name='fconv4')
        #    self.regressor = tf.layers.conv2d(f8, filters=4, kernel_size=1,
        #            strides=1, name='regress_out')
        # training
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.squeeze_logits, labels=self.Y1,name='loss'))
        self.num_correct = tf.equal(tf.argmax(tf.nn.softmax(self.squeeze_logits), 1),
                tf.argmax(self.Y1,1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(self.num_correct, tf.float32),
                name='accuracy')
        self.opt = tf.train.AdamOptimizer() # default
        self.minimizer = self.opt.minimize(self.loss, name='minimizer')
    


class BDDModel():
    """
    A classification head model for 10-class classification on the BDD100k
    dataset. Modeled loosely after the Overfeat "fast" architecture 
    Not actually used, just for learning

    """
    def __init__(self):
        """
        Constructs the model
        Input is assumed to be 256 by 256 by 3

        """
        
        self.X =  tf.placeholder(tf.float32, shape=(None, 256, 256, 3))
        self.Y1 = tf.placeholder(tf.float32, shape=(None, 10))  # 10 class
        self.Y2 = tf.placeholder(tf.float32, shape=(None, 4)) # regression bbox

        # Layer 1: Conv + MaxPool
        c1 = tf.layers.conv2d(self.X, filters=96, kernel_size=11, strides=3,
               name='conv1')
        p1 = tf.layers.max_pooling2d(c1, pool_size=2, strides=2,
               name='maxpool1')

        # Layer 2: Conv + MaxPool
        c2 = tf.layers.conv2d(p1, filters=256, kernel_size=4, strides=1,
               name='conv2')
        p2 = tf.layers.max_pooling2d(c2, pool_size=2, strides=2,
               name='maxpool2')

        # Layer 3: Conv
        c3 = tf.layers.conv2d(p2, filters=512, kernel_size=4, strides=1,
               name='conv3')

        # Layer 4: Conv + MaxPool
        c4 = tf.layers.conv2d(c3, filters=1024, kernel_size=3, strides=1,
               name='conv5')
        p4 = tf.layers.max_pooling2d(c4, pool_size=2, strides=2,
               name='maxpool5')

        # "FC" Conv layers CLASSIFICATION HEAD
        # output of p4: 7 by 7 by 1024
        f5 = tf.layers.conv2d(p4, filters=3072, kernel_size=7, strides=1,
               name='fconv1')
        f6 = tf.layers.conv2d(f5, filters=4096, kernel_size=1, strides=1,
               name='fconv2')
        self.logits = tf.layers.conv2d(f6, filters=10, kernel_size=1,strides=1,
               name='fconvout')

        
        # training
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
           labels=self.Y1,name='loss'))
        self.opt = tf.train.AdamOptimizer() # default
        self.minimizer = self.opt.minimize(self.loss, name='minimizer')
        
