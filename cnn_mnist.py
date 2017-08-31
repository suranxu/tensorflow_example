from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import sys
import numpy as np
import tensorflow as tf  

class Config(object):
	def __init__(self):
		self.trainFlag = False
		self.restore_from_checkpoint = False
		self.learning_rate = 0.1
		self.batch_size = 100
		self.display_step = 1
		self.training_epochs = 10
		self.dropout = 0.7

		if not self.trainFlag:
			self.restore_from_checkpoint = True
			self.dropout = 1.0

class CNNModel(object):
	def __init__(self, conf):
		#Netwrok parameters:
		self.n_input = 784 # 28 * 28 pixel
		self.n_classes = 10

		#place holder
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.dropout = tf.placeholder_with_default(1.0, shape=())

		# variables
		self.weights = {
			# window size 5*5, kernal 32
			'wc_1' : tf.get_variable("wc_1", [5, 5, 1, 32]), 
			'wc_2' : tf.get_variable("wc_2", [5, 5, 32, 64]),
			'wf_1' : tf.get_variable("wf_1", [7*7*64, 1024]),
			'w_out' : tf.get_variable("w_out", [1024, self.n_classes]) 
		}

		self.biases = {
			'bc_1' : tf.get_variable("bc_1", [32]),
			'bc_2' : tf.get_variable("bc_2", [64]),
			'bf_1' : tf.get_variable("bf_1", [1024]),
			'b_out': tf.get_variable("b_out", [self.n_classes])
		}

		self.cx = tf.reshape(self.x, shape=[-1, 28, 28, 1]) #[batch, 28, 28, 1]
		self.layer1 = tf.nn.conv2d(self.cx, self.weights['wc_1'], strides=[1,1,1,1], padding='SAME') #[batch, 28, 28, 32]
		self.layer1 = tf.nn.bias_add(self.layer1, self.biases['bc_1']) #[batch, 28, 28, 32]
		self.layer1 = tf.nn.relu(self.layer1)
		self.layer1 = tf.nn.max_pool(self.layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #[batch, 14, 14, 32]

		self.layer2 = tf.nn.conv2d(self.layer1, self.weights['wc_2'], [1,1,1,1], padding='SAME') #[batch, 14, 14, 64]
		self.layer2 = tf.nn.bias_add(self.layer2, self.biases['bc_2'])
		self.layer2 = tf.nn.relu(self.layer2)
		self.layer2 = tf.nn.max_pool(self.layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #[batch, 7, 7, 64]
		
		self.layer2 = tf.reshape(self.layer2, [-1, 7*7*64]) #[batch, 7*7*64]
		self.layer3 = tf.nn.xw_plus_b(self.layer2, self.weights['wf_1'], self.biases['bf_1']) #[batch, 1024]
		self.layer3 = tf.nn.relu(self.layer3)
		self.layer3 = tf.nn.dropout(self.layer3, self.dropout)

		self.out = tf.nn.xw_plus_b(self.layer3, self.weights['w_out'], self.biases['b_out']) #[batch, classes]
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.y))
		self.optimizer = tf.train.GradientDescentOptimizer(conf.learning_rate).minimize(self.cost) 

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1)) , "float") )

if __name__ == "__main__":

	conf = Config()
	with tf.Graph().as_default():
		np.random.seed(0)
		tf.set_random_seed(0)
		with tf.Session() as sess:
			
			with tf.variable_scope("CNNMinstModel", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
				m = CNNModel(conf)
			saver = tf.train.Saver()
			
			sess.run(tf.global_variables_initializer())
			
			if conf.restore_from_checkpoint:
				newSaver = tf.train.import_meta_graph("mnistCNNModel.dat.meta")
				newSaver.restore(sess, tf.train.latest_checkpoint('./'))
			
			if conf.trainFlag:
				#get data, start training
				for i in range(conf.training_epochs):
					avgacc = 0.0;
					total_batch = int(mnist.train.num_examples / conf.batch_size)
					for j in range(total_batch):
						batch_x, batch_y = mnist.train.next_batch(conf.batch_size)
						_, acc = sess.run([m.optimizer, m.accuracy], feed_dict={m.x:batch_x, m.y:batch_y, m.dropout:conf.dropout})
						avgacc += acc / total_batch
					print("epoch %d:" % i, "accuracy=%.9f" % avgacc)
					saver.save(sess, "mnistCNNModel.dat")
			else:
				acc = sess.run(m.accuracy, feed_dict={m.x:mnist.test.images, m.y: mnist.test.labels, m.dropout:1.0})
				print ("test result, accuracy=%.9f" % acc)





