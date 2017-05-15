import tensorflow as tf
import check_mat
from tensorflow.contrib import rnn

action_label = 'throw'
full_action_dict, actions_pruned = check_mat.save_actions('joint_positions')
#fetch action data depending on the action label
def fetch_action_data(action_label, full_action_dict, actions_pruned):
	return [full_action_dict[action_label][i] for i in actions_pruned[action_label]]

data = fetch_action_data(action_label, full_action_dict, actions_pruned)
framerate = len(data[0]['scale'][0])
print("framerate looks like", framerate)

for action in full_action_dict.keys():
	print("For action ", action, ", # of right elements is ", len(actions_pruned[action]))

def set_model(action_type):

	# Parameters
	learning_rate = 0.001
	training_iters = 100000
	batch_size = 128
	display_step = 10

	# Network Parameters
	n_input = 15 # Joint numbers
	n_steps = 40 # timesteps (framerate)
	n_hidden = 128 # hidden layer num of features
	n_classes = 10 # MNIST total classes (0-9 digits)

	# tf Graph input
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("float", [None, n_classes])

	# Define weights
	weights = {
	    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	}
	biases = {
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}


	def RNN(x, weights, biases):

	    # Prepare data shape to match `rnn` function requirements
	    # Current data input shape: (batch_size, n_steps, n_input)
	    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
	    
	    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	    x = tf.unstack(x, n_steps, 1)

	    # Define a lstm cell with tensorflow
	    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	    # Get lstm cell output
	    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	    # Linear activation, using rnn inner loop last output
	    return tf.matmul(outputs[-1], weights['out']) + biases['out']

	pred = RNN(x, weights, biases)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables
	init = tf.global_variables_initializer()
