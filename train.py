import tensorflow as tf
import numpy as np  
import check_mat
import model


def train(model_lstm, batches, num_iterations, print_every, save_dir):
	#to save with
	saver = tf.train.Saver()

	with tf.Session() as sess:
		for i in range(num_iterations):
			sess.run(tf.global_variables_initializer())
			values = {
				model_lstm.input_data: batches[i]
			}
			train_loss, state, W, b = \
			sess.run([model_lstm.cost, model_lstm.final_states, model_lstm.W, model_lstm.b], values)
			if i % print_every == 0: #print every
				print("epoch {}, train_loss = {:.3f}".format(i, train_loss))

		save_path = saver.save(sess, save_dir + "/model.ckpt")


def load_data(data_dir, action_label):
	full_action_data, actions_pruned = check_mat.save_actions(data_dir)
	action_data = check_mat.fetch_action_data(action_label, full_action_data, actions_pruned)
	T = action_data[0]['pos_world'].shape[2]
	return action_data, T

def split_data(action_data, num_iterations, num_batches, T):
	#arrange input data as N*30*T
	batches = []
	for i in range(num_iterations):
		chosen_indices = np.random.choice(len(action_data), num_batches)
		#print("chosen_indices is ", chosen_indices)
		batches.append(np.vstack(tuple([\
			np.reshape(action_data[i]['pos_world'], (1, -1, T))  for i in chosen_indices])))

	return batches



data_dir = 'data/joint_positions'
action_label = 'shoot_bow'
action_data, T = load_data(data_dir, action_label)

#Various model arguments	
learning_rate = 0.1
num_batches = 5
num_iterations = 1000
state_size = 50
layer_num = 3
domain_size = 100

#define model
model_lstm = model.Model(\
	learning_rate, num_batches, T, state_size, layer_num)

#split data into batches
batches = split_data(action_data, num_iterations, num_batches, T)

#train params
print_every = 100
save_dir = 'data'
train(model_lstm, batches, num_iterations, print_every, save_dir)
