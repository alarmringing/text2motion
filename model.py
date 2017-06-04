import tensorflow as tf
import numpy as np  
from tensorflow.contrib import rnn


class Model():
	def __init__(self, learning_rate, batch_size, T, H_size, layer_num):

		#input is a N*30*40 vector, where
		#N: number of batches
		#30: x,y positions for each 15 joints
		#T: timestep
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.T = T
		self.H_size = H_size
		self.layer_num = layer_num
		self.model_num = 30 #THIS IS FIXED! NUMBER OF MODELS, EACH 1D TIMELINE (BATCH * LENGTH)

		#Tensorflow only supports 1D LSTM (for now), so use this
		self.models = []

		# Global config variables
		pos = ['x','y']

		#to return

		#placeholders
		self.input_data = tf.placeholder(tf.float32, [self.batch_size, self.model_num, self.T])

		#start building model for each joint
		
		#for i in range(self.model_num):
		#for each joint and mark wheter x or y
		#with tf.variable_scope('joint_'+str(i/2)+'_'+pos[i%2]):

		#(slice of) placeholders for input and output	
		input = tf.unstack(self.input_data[:, :, :-1], axis=2)
		target = tf.unstack(self.input_data[:, :, 1:], axis=2)

		#attach multiple LSTM cells depending on layer_num
		rnn_cells = []
		for j in range(self.layer_num):
			rnn_cells.append(rnn.BasicLSTMCell(self.H_size))
		with tf.variable_scope('lstm'):
			rnn_cell = rnn.MultiRNNCell(rnn_cells)

		#initialize state state
		initial_state = rnn_cell.zero_state(self.H_size, tf.float32) #initialize weights

		#logits and predictions
		with tf.variable_scope('output'):
			#state output from rnn
			outputs, states = rnn.static_rnn(rnn_cell, input, dtype=tf.float32)
			self.final_states = states

		with tf.variable_scope('softmax'):
			self.W = tf.get_variable('W', [self.H_size, self.model_num])
			self.b = tf.get_variable('b', [self.model_num], initializer=tf.constant_initializer(0.0))

			pred = []
			for output in outputs: 
				pred.append(tf.matmul(output, self.W) + self.b)

			#self.costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=target))
			self.cost = tf.reduce_sum(tf.pow(tf.stack(pred)-tf.stack(target), 2))/(2 * self.model_num)

		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			optimizer.minimize(self.cost)


