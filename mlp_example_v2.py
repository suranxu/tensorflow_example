from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import sys
import numpy as np
import tensorflow as tf  

#todo
# put all model structure to class Model
# use __main__

# Parameters
class Config(object):
	def __init__(self):
		self.trainFlag = True
		self.restore_from_checkpoint = True
		self.learning_rate = 0.1
		self.batch_size = 100
		self.display_step = 1
		self.training_epochs = 10

		if not self.trainFlag:
			self.restore_from_checkpoint = True

class MLPModel(object):
	def __init__(self, config):
		self.n_hidden_1 = 256
		self.n_hidden_2 = 256
		self.n_input = 784
		self.n_classes = 10

		# placeholder
		self.x = tf.placeholder("float", [None, self.n_input])
		self.y = tf.placeholder("float", [None, self.n_classes])

		# variables
		self.weights = {
			'h1' : tf.get_variable("h1_w", [self.n_input, self.n_hidden_1]),
			'h2' : tf.get_variable("h2_w", [self.n_hidden_1, self.n_hidden_2]),
			'h3' : tf.get_variable("h3_w", [self.n_hidden_2, self.n_classes])
		}
		self.biases = {
			'h1' : tf.get_variable("h1_b", [self.n_hidden_1]),
			'h2' : tf.get_variable("h2_b", [self.n_hidden_2]),
			'h3' : tf.get_variable("h3_b", [self.n_classes])
		}

		with tf.device('/cpu:0'):
			print ("start forward computing...")
			self.layer_1 = tf.nn.relu(tf.nn.xw_plus_b(self.x, self.weights['h1'], self.biases['h1']))
			self.layer_2 = tf.nn.relu(tf.nn.xw_plus_b(self.layer_1, self.weights['h2'], self.biases['h2']))
			self.output = tf.nn.xw_plus_b(self.layer_2, self.weights['h3'], self.biases['h3'])

			# calc cost
			global_step = tf.Variable(0, name="global_step_cnn", trainable=False)
			self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y)
			self.optimizer = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.cost, global_step=global_step)
			self.avgcost = tf.reduce_mean(self.cost)
			# for test
			self.correct_prediction = tf.equal( tf.argmax(self.output, 1), tf.argmax(self.y, 1) )
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) 

if __name__ == "__main__":
	# random seed fixed
	np.random.seed(0)
	tf.set_random_seed(0)

	#init
	conf = Config()


	# Lauch the graph
	with tf.Graph().as_default():
		with tf.Session() as sess:
			with tf.variable_scope("model", reuse=None, initializer=tf.contrib.layers.xavier_initializer() ) : 
				m = MLPModel(config = conf)
			
			#Note: you need to init saver before restore action 
			saver = tf.train.Saver()

			# init variable, before restore
			sess.run(tf.global_variables_initializer())

			# reload last model
			if conf.restore_from_checkpoint:
				new_saver = tf.train.import_meta_graph('mlp_parameters2.dat.meta')
				new_saver.restore(sess, tf.train.latest_checkpoint('./'))

			# train or test
			if conf.trainFlag:
				#Train model
				
				for epoch in range(conf.training_epochs):
					avg_cost = 0.0
					total_batch = int(mnist.train.num_examples / conf.batch_size)
					# Run one batch
					for i in range(total_batch):
						batch_x, batch_y = mnist.train.next_batch(conf.batch_size)
						_, c, acu = sess.run([m.optimizer, m.avgcost, m.accuracy], feed_dict={m.x:batch_x, m.y:batch_y})
						avg_cost += c / total_batch
					if epoch % conf.display_step == 0:
						print("epoch %04d" % epoch, "avgcost=%.9f" % avg_cost, "batchAccuracy=%.9f" % acu)
				print("Optimization Finished")
				saver.save(sess, "mlp_parameters2.dat")
				accu = sess.run(m.accuracy, feed_dict={m.x:mnist.train.images, m.y:mnist.train.labels})
				print ("Overall accuracy on training set:%.9f" % accu)
				accu = sess.run(m.accuracy, feed_dict={m.x : mnist.test.images, m.y : mnist.test.labels})
				print ("Overall accuracy on testing set:%.9f" % accu)
			else:
				#Test Model
				accu = sess.run(m.accuracy, feed_dict={m.x : mnist.test.images, m.y : mnist.test.labels})
				print ("Overall accuracy:%.9f" % accu)
		


