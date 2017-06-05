import tensorflow as tf
import numpy as np  
from tensorflow.contrib import rnn


class Model():
	def __init__(self, learning_rate, batch_size, T, H_size, layer_num):

		#input is a N*30*40 vector, where
		#N: number of batches
		#30: x,y positions for each 15 joints
		#T: timestep
		
		self.initial_learning_rate = learning_rate
		global_step = tf.Variable(0, trainable = False)
		learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step, 1000, 0.995, staircase=True)
		self.batch_size = batch_size
		self.H_size = H_size
		self.layer_num = layer_num
		self.model_num = 30 #THIS IS FIXED! NUMBER OF MODELS, EACH 1D TIMELINE (BATCH * LENGTH)

		#placeholders
		self.input_data = tf.placeholder(tf.float32, [None, self.model_num, T], name = 'input_data')

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

		with tf.variable_scope('regression'):

			self.pred = []
			W = tf.get_variable('W', [self.H_size, self.model_num])
			b = tf.get_variable('b', [self.model_num], initializer = tf.constant_initializer(0.0))
			for k in range(len(outputs)):
				output = outputs[k] 
				self.pred.append(tf.matmul(output, W) + b)

			self.pred = tf.stack(self.pred, name = 'predictions')

			self.cost = tf.reduce_sum(tf.pow(self.pred-tf.stack(target), 2)/self.model_num, name = 'cost')
			#check gradient
			self.var_grad = tf.gradients(self.cost, [outputs[0]])[0]

		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(learning_rate)
			self.updates = optimizer.minimize(self.cost, global_step = global_step)



