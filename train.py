import tensorflow as tf
import numpy as np  
import check_mat
import model


def train(model_lstm, batches, num_iterations, print_every, save_dir):
	#to save with
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_iterations):
			values = {
				model_lstm.input_data: batches[i]
			}
			_, train_loss, state, gradients = sess.run([model_lstm.updates, model_lstm.cost, model_lstm.final_states, model_lstm.var_grad], values)
			if i % print_every == 0: #print every
				print("epoch {}, train_loss = {:.3f}".format(i, train_loss))

		save_path = saver.save(sess, save_dir + "/model_lstm")


def load_data(data_dir, action_label):
	full_action_data, actions_pruned = check_mat.save_actions(data_dir)
	action_data = check_mat.fetch_action_data(action_label, full_action_data, actions_pruned)
	T = action_data[0]['pos_world'].shape[2]
	return action_data, T

def split_data(test_num, val_num, action_data, num_iterations, num_batches, T):
	
	available_indices = range(len(action_data))

	#randomly select test and val data
	np.random.shuffle(available_indices)
	test_indices = [available_indices.pop() for i in range(test_num)]
	val_indices = [available_indices.pop() for i in range(val_num)]
	test = [action_data[i] for i in test_indices]
	val = [action_data[i] for i in val_indices]

	#write to batches list
	batches = []
	for i in range(num_iterations):
		#shapes data to N * 30 * T
		chosen_indices = np.random.choice(available_indices, num_batches)
		batches.append(np.vstack(tuple([\
			np.reshape(action_data[i]['pos_world'], (1, -1, T))  for i in chosen_indices])))


	return batches, val, test



if __name__ == '__main__':

	data_dir = 'data/joint_positions'
	action_label = 'shoot_bow'
	action_data, T = load_data(data_dir, action_label)

	#Various model arguments	
	learning_rate = 1e-5
	num_batches = 5
	num_iterations = 50000
	state_size = 50
	layer_num = 3
	domain_size = 100

	#define model
	model_lstm = model.Model(\
		learning_rate, num_batches, T, state_size, layer_num)

	#split data into batches
	test_num = 2
	val_num = 2
	batches, val, test = split_data(test_num, val_num, action_data, num_iterations, num_batches, T)

	#train params
	print_every = 5000
	save_dir = 'data'
	train(model_lstm, batches, num_iterations, print_every, save_dir)
