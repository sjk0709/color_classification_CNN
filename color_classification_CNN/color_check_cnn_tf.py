# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:56:53 2017

@author: Jaekyung
"""
import sys, os
sys.path.append(os.pardir)  # parent directory
import tensorflow as tf
import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from sklearn.feature_extraction import image
#from PIL import Image
#import glob
#import random


#import galaxy_JK
import data_generation_JK

from tensorflow.python.tools import freeze_graph
from tensorflow.examples.tutorials.mnist import input_data
#
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#

tf.set_random_seed(777) # reproducibility


class Model:
    
    def __init__(self, sess, name, learning_rate=0.0001, feature_shape=[28,28,1], lable_size=10, weight_decay_rate=1e-5, withScatter=True):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._feature_shape = feature_shape
        self.lable_size = lable_size
        self._withScatter = withScatter
        
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay_rate)
        
        self._build_net()
        
        
        
    def _build_net(self):        
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing            
            self.training = tf.placeholder(tf.bool, name="training")

            # input place holders
            self.X = tf.placeholder( tf.float32, [None, self._feature_shape[0]*self._feature_shape[1]*self._feature_shape[2]], name="input")           
            X_img = tf.reshape(self.X, [-1, self._feature_shape[0], self._feature_shape[1], self._feature_shape[2]])
            self.Y = tf.placeholder(tf.float32, [None, self.lable_size])
            
            # Convolutional Layer #1 and # Pooling Layer #1
            conv11 = tf.layers.conv2d(inputs=X_img, filters=64, kernel_size=[3,3], 
                                     padding="SAME", activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,60,60,64)

            conv12 = tf.layers.conv2d(inputs=conv11, filters=64, kernel_size=[3,3], 
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,60,60,64)
            
            pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2,2],
                                            padding="SAME", strides=2)              # (?,14,14,64)  # (?,16,16,64)32
            
            dropout1 = tf.layers.dropout(inputs=pool1, 
                                         rate=0.7, training=self.training)
            
            # Convolutional Layer #2 and Pooling Layer #2
            conv21 = tf.layers.conv2d(inputs=dropout1, filters=128, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,30,30,64)
            
            conv22 = tf.layers.conv2d(inputs=conv21, filters=128, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,30,30,64)
            
            conv23 = tf.layers.conv2d(inputs=conv22, filters=128, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,15,15,64)
            
            pool2 = tf.layers.max_pooling2d(inputs=conv23, pool_size=[2,2],
                                            padding="SAME", strides=2)              # (?,7,7,64)  # (?,8,8,64)16
            
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)
            
            # Convolutional Layer #2 and Pooling Layer #3
            conv31 = tf.layers.conv2d(inputs=dropout2, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,15,15,256)
            
            conv32 = tf.layers.conv2d(inputs=conv31, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,15,15,256)
            
            conv33 = tf.layers.conv2d(inputs=conv32, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,15,15,256)
            
            conv34 = tf.layers.conv2d(inputs=conv33, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)         # (?,15,15,256)
            
            pool3 = tf.layers.max_pooling2d(inputs=conv34, pool_size=[2,2],
                                            padding="SAME", strides=2)              # (?,4,4,256)   # (?,4,4,256)8
            
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)
            
            # Dense Layer with Relu ========================================================================
            flat = tf.reshape(dropout3, [-1, 256*8*8])                              # 32-(?,4*4*128)  # 60-(?,8*8*128) 
                              
            dense4 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)              # (?,1024)
            
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
            
            dense5 = tf.layers.dense(inputs=dropout4, units=1024, activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)              # (?,1024)
            
            dropout5 = tf.layers.dropout(inputs=dense5, rate=0.5, training=self.training)
 
            self.logits = tf.layers.dense(inputs=dropout5, units=self.lable_size, 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          kernel_regularizer=self.kernel_regularizer )                # (?,2)
            
                    
        self.prob = tf.nn.softmax(self.logits, name="prob")
        self.result = tf.argmax(self.logits, 1, name="result")
            
        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.prob,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

class Commander :
    
    def __init__(self, Model, data, feature_shape=[28,28,1], label_size=2, learning_rate=0.0001, training=False, 
                 color_type="Lab", withScatter=True, feature_type='full'):
        
        self._modelName = "G8A" 
        pre_path = self._modelName + "-" + color_type
        if(withScatter):
            pre_path = pre_path + 'withScatter'
        pre_path = "../../" + pre_path + '-'+ str(feature_shape[0]) +'x'+str(feature_shape[1]) + feature_type
            
        self.checkpoint_state_name = "checkpoint_state"
        self.saved_checkpoint = 'saved_checkpoint'
        self.input_graph_name = "input_graph.pb"               
        
        
        # initialize
        self.sess = tf.Session()
        self.data = data
    
        self.label_size = label_size
   
        self._model = Model(self.sess, self._modelName, learning_rate=learning_rate, 
                                     feature_shape=feature_shape, lable_size=self.label_size, withScatter=withScatter )
        
#        if not os.path.exists('out/'):
#            os.makedirs('out/')
        #dir_path = os.path.dirname(os.path.realpath(__file__))  # To get the full path to the dirctory
        #cwd = os.getcwd()                                       # To get the current working directory
        
            
        self.model_dir = pre_path + "/"
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
    
        self.checkpoint_dir = pre_path  + "/parameters" + "/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
               
        #self.checkpoint_prefix = os.path.join(self.checkpoint_dir, '/ensemble_model'+str(self.start_model_num)+'_'+str(self.end_model_num))
        #print(self.checkpoint_prefix)        
       
        self.input_graph_path = self.model_dir + self.input_graph_name        
          
        self.checkpoint_prefix = self.checkpoint_dir + self.saved_checkpoint
        self.input_checkpoint_path = self.checkpoint_prefix + "-0"  

        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir, latest_filename=self.checkpoint_state_name)

        self.sess.run(tf.global_variables_initializer())
        
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif False:
            raise Exception("Could not load checkpoints for playback")
        else:
            print("Frist training")
            
    def train(self, training_epochs=20, batch_size=128):
        # Save our model
        tf.train.write_graph(self.sess.graph_def, self.model_dir, self.input_graph_name, as_text=True)
        
        # train my model
        print('Learning Started!')
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(self.data.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = self.data.train.next_batch(batch_size)
#                print(batch_xs.shape, batch_ys.shape)
                # train each model                
                cost, _ = self._model.train(batch_xs, batch_ys)
                avg_cost += cost            
    
            avg_cost /= total_batch 
            # save parameters    
#            save_path = saver.save(sess, checkpoint_path + '/network')
            save_path = self._saver.save(self.sess, self.checkpoint_prefix, global_step=0, latest_filename=self.checkpoint_state_name)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost)
 
        # show all variables name
#        for op in tf.get_default_graph().get_operations():
#            print (str(op.name))
                
        # Save our model
        tf.train.write_graph(self.sess.graph_def, self.model_dir, self.input_graph_name, as_text=True)
        print('Learning Finished!')   
        
    def test(self):
        # Test model and check accuracy        
        test_size = 100 # len(self.data.test.labels)     
        testX, testY = self.data.test.next_batch(test_size)        
        
        print('logits : ', self._model.predict(testX))     
        print('Accuracy:', self._model.get_accuracy( testX, testY ))
        
    
    def freezeModel(self, output_node_names="prob", output_graph_name="output_graph.pb" ):                   
        # Note that we this normally should be only "output_node"!!!
        input_saver_def_path = "" 
        input_binary = False                        
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"    
        output_graph_path = self.model_dir + output_graph_name
        clear_devices = False        
        freeze_graph.freeze_graph(self.input_graph_path, input_saver_def_path,
                                  input_binary, self.input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_path,
                                  clear_devices, False)
        print('Freezing the model finished!')
            
if __name__ == '__main__':  
       
    mode = int(input("1.training  |  2.accuracy test  |  3.Freeze a model  : "))
    
    dataPath = '../../../color_defect_galaxy/data/train_data'
    color_type = 'Lab'  #  RGB   /  BGR   /   Lab   /   Gray
    feature_type = 'block'  #  full   /   block
    withScatter = False

    sample_size = [1000, 400]    
    feature_shape = [64, 64, 3]
    
        
    if( color_type=='Gray'):
        feature_shape[2] = 1
        feature_type = 'full'
    
    if( withScatter):
        feature_shape[2] = feature_shape[2]+1
    
    if(feature_type == 'full'):
        sample_size[0] = feature_shape[0]
        sample_size[1] = feature_shape[1]
        
        
    label_size = 2                  # OK=[1 0] or NG=[0 1]
    learning_rate=1e-4
    
    training_epochs = 1000
    batch_size = 20
    nExtractPerImg = 50
    nFakes = 100     # if you don't want to generate fake date, nFakes = 0
    
    # Pre process
    if(mode==1 or mode==2):        	
#        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
        galaxy = data_generation_JK.Galaxy (dataPath, sampleSize=sample_size, feature_shape=feature_shape, color_type=color_type, feature_type=feature_type,
                                            nExtract=nExtractPerImg, withScatter=withScatter, nFakes=nFakes)  
#        galaxy = Galaxy (dataPath, data_shape=feature_shape, feature_type=feature_type)
        mean, std = galaxy.getMeanStd()
        commander = Commander(Model=Model,  
                            data=galaxy, feature_shape=feature_shape, label_size=label_size, 
                            learning_rate=learning_rate, color_type=color_type, withScatter=withScatter, feature_type=feature_type)
        
    elif(mode==3 ):
        commander = Commander(Model=Model,  
                            data=None, feature_shape=feature_shape, label_size=label_size, 
                            learning_rate=learning_rate, color_type=color_type, withScatter=withScatter, feature_type=feature_type)
        
    # training or test
    if (mode==1):
        commander.train(training_epochs=training_epochs, batch_size=batch_size)
        commander.freezeModel()
        
    elif (mode==2):
        commander.test()        

    elif (mode==3):
        commander.freezeModel()
    

