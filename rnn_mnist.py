#
# training set overall accuracy 0.955872715
# testing set overall accuracy 0.955699980
#
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import sys
import numpy as np
import tensorflow as tf  

class Config(object):
	def __init__(self):
		#training time control
		self.training_epoches = 10
		self.learning_rate = 0.2
		self.batch_size = 1000

		# train / test
		self.train_flag = False
		self.restore_from_checkpoint = False

		# other parameters
		self.display_step = 1
		self.dropout_keep_prob = 0.5

		if not self.train_flag:
			self.restore_from_checkpoint = True
			self.dropout_keep_prob = 1.0

class LSTMModel(object):
	def __init__(self, conf):
		#define net size 
		self.num_input = 28 #one line of image
		self.timesteps = 28 #28 lines
		self.num_hidden = 128 #RNN embedding size
		self.num_classes = 10 #target size

		#mnist seq len is fixed, in real life this should be a placeholder, seq len is associate with each batch
		self.sequence_length = tf.fill([conf.batch_size], self.timesteps)

		self.weights={
			'out' : tf.get_variable("w_out", [self.num_hidden, self.num_classes])
		}
		self.biases = {
			'out' : tf.get_variable("b_out", [self.num_classes])
		}

		#define placeholders
		#self.x = tf.placeholder("float", [None, self.timesteps, self.num_input]) # [batchsize, 28, 28]
		self.x = tf.placeholder("float", [None, self.timesteps * self.num_input]) # [batchsize, 28 * 28]
		self.y = tf.placeholder("float", [None, self.num_classes])
		self.dropout = tf.placeholder_with_default(1.0, shape=())

		print(tf.shape(self.x), tf.shape(self.y), tf.shape(self.sequence_length))

		#forward network
		self.x4rnn = tf.reshape(self.x, [-1, self.timesteps, self.num_input])
		print(tf.shape(self.x4rnn))
		self.x4rnn = tf.unstack(self.x4rnn, axis=1) #List of size:timesteps:28, each element is [batchsize, num_input:28]

		self.lstm_cell = tf.contrib.rnn.DropoutWrapper( tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0), output_keep_prob=self.dropout) 
		#tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(first_dim), output_keep_prob = config.dropout_keep_prob)

		self.outputs, self.states = tf.contrib.rnn.static_rnn(self.lstm_cell, self.x4rnn, dtype=tf.float32, sequence_length=self.sequence_length) 
		#outputs is a list of size timesteps, each element is a num_hidden embedding
		self.out = tf.nn.xw_plus_b(self.outputs[-1], self.weights['out'], self.biases['out'])

		#loss and feedback
		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.y)
		self.avgloss = tf.reduce_mean(self.loss)
		self.optimizer = tf.train.GradientDescentOptimizer(conf.learning_rate).minimize(self.avgloss)

		#for test
		self.accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(self.out, 1), tf.argmax(self.y, 1) ), "float") )

if __name__ == "__main__":
	#init before graph
	np.random.seed = 0
	tf.set_random_seed(0)

	with tf.Graph().as_default():
		with tf.Session() as sess:
			conf = Config()
			with tf.variable_scope("LSTMModel", reuse=None, initializer=tf.contrib.layers.xavier_initializer()):
				m = LSTMModel(conf)

			#init saver before restore
			saver = tf.train.Saver()

			sess.run(tf.global_variables_initializer())

			if conf.restore_from_checkpoint:
				new_saver = tf.train.import_meta_graph('lstm_parameters.dat.meta')
				new_saver.restore(sess, tf.train.latest_checkpoint('./'))

			if conf.train_flag:
				train_steps = int(mnist.train.num_examples / conf.batch_size)
				for i in range(conf.training_epoches):
					avgloss = 0.0;
					avgacc = 0.0;
					for j in range(train_steps):
						batch_x,batch_y = mnist.train.next_batch(conf.batch_size)
						_,loss,a = sess.run([m.optimizer, m.avgloss, m.accuracy], feed_dict={m.x:batch_x, m.y:batch_y, m.dropout:conf.dropout_keep_prob})
						avgloss += loss / train_steps;
						avgacc += a / train_steps;
					if i % conf.display_step == 0:
						print("epoch %4d"%i, "avgloss=%.9f"%avgloss, "avgacc=%.9f"%avgacc)
					saver.save(sess, "lstm_parameters.dat")
				print("training set overall accuracy %.9f" % sess.run(m.accuracy, feed_dict={m.x:mnist.train.images, m.y:mnist.train.labels, m.dropout: 1.0}) )
				print("testing set overall accuracy %.9f" % sess.run(m.accuracy, feed_dict={m.x:mnist.test.images, m.y:mnist.test.labels, m.dropout:1.0}))
			else:
				print("testing set overall accuracy %.9f" % sess.run(m.accuracy, feed_dict={m.x:mnist.test.images, m.y:mnist.test.labels, m.dropout:1.0}))



