import tensorflow as tf
import numpy as np 
import check_mat
import model
import random
import os
import time


def train(model_lstm, batches, num_iterations, print_every, save_every, save_dir, save_name):
	
	#to save with
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		summaries = tf.summary.merge_all()
		writer = tf.summary.FileWriter(os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S")))
		writer.add_graph(sess.graph)
		sess.run(tf.global_variables_initializer())

		for i in range(num_iterations):
			start = time.time()
			values = {
				model_lstm.input_data: batches[i]
			}

			_, train_loss, state, \
			target, pred, gradients = sess.run([model_lstm.updates, model_lstm.cost, model_lstm.final_states, model_lstm.target, model_lstm.pred, model_lstm.var_grad], values)

			#instrument for tensorboard	
			summ, _, train_loss, state, gradients = sess.run([summaries, model_lstm.updates, model_lstm.cost, model_lstm.final_states, model_lstm.var_grad], values)
			writer.add_summary(summ, i*num_batches)


			if i % print_every == 0: #print every
				save_path = saver.save(sess, save_dir + "/" + save_name)
				print("epoch {}/{}, train_loss = {:.7f}".format(i, num_iterations, train_loss))
				print("target is ", target[0][0,:], " and pred is ", pred[0,0,:])
			end = time.time()
			
		save_path = saver.save(sess, save_dir + "/" + save_name)


def load_data(data_dir, action_label):
	full_action_data, actions_pruned = check_mat.save_actions(data_dir)
	action_data = check_mat.fetch_action_data(action_label, full_action_data, actions_pruned)
	T = action_data[0]['pos_world'].shape[2]
	return action_data, T

def split_data(test_num, val_num, action_data, num_iterations, num_batches, T):
	
	available_indices = list(range(len(action_data)))

	#randomly select test and val data
	random.shuffle(available_indices)
	test_indices = [available_indices.pop() for i in range(test_num)]
	val_indices = [available_indices.pop() for i in range(val_num)]
	print("test_indices are ", test_indices, " and val_indices are ", val_indices)
	test = [action_data[i] for i in test_indices]
	val = [action_data[i] for i in val_indices]

	#write to batches list
	batches = []
	for i in range(num_iterations):
		#shapes data to N * 30 * T
		chosen_indices = np.random.choice(available_indices, num_batches)
		chosen_mats = []
		for j in chosen_indices:
			stacked = np.vstack((action_data[j]['pos_world'][0], action_data[j]['pos_world'][1])) #30*T
			stacked = np.expand_dims(stacked, axis=0) #turns it into 1*30*T
			chosen_mats.append(stacked)

		batches.append(np.vstack(tuple(chosen_mats)))

	return batches, val, test



if __name__ == '__main__':

	data_dir = 'data/joint_positions'
	action_label = 'shoot_bow'
	action_data, T = load_data(data_dir, action_label)

	#Various model arguments	
	learning_rate = 1.5e-3
	num_batches = 10
	num_iterations = 50000
	state_size = 100
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
	print_every = 1000
	save_every = 10000
	save_dir = 'data'
	save_name = 'model_lstm_test'
	train(model_lstm, batches, num_iterations, print_every, save_every, save_dir, save_name)
