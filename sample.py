import tensorflow as tf
import numpy as np 
import train
import animation

def sample(model_path, initial_input):
	
	def mean_squared_error(pred, truth):
		return np.mean((pred - truth)**2)/(pred.size)

	with tf.Session() as sess:
		#restores the model 
		saver = tf.train.import_meta_graph(model_path + 'model_lstm.meta')
		saver.restore(sess, model_path + 'model_lstm')
		#initialize session and graph
		sess.run(tf.global_variables_initializer())
		graph = tf.get_default_graph()
		T = initial_input.shape[2]
		pred_result = np.zeros((2, 15, T-1)) #this is fixed

		#at each timestep
		initial_timestep = initial_input #full ground truth information
		for t in range(initial_input.shape[2]-1):
			values = {
				"input_data:0": initial_timestep
			}
			predictions = sess.run([graph.get_tensor_by_name("regression/predictions:0")], values)

			this_pred = predictions[0][0, 0, :]
			pred_result[0, :, t] = this_pred[:15]
			pred_result[1, :, t] = this_pred[15:]
			initial_timestep = np.concatenate((np.reshape(this_pred, (1,-1,1)), np.zeros((1,30,T-1))), axis=2) #second value is meaningless
			
			#report error with ground truth in this timestep
			print("timestep: ", t, " error: ", mean_squared_error(initial_input[:,:,t+1].flatten(), predictions[0][t,0,:].flatten()))
		return pred_result

def generate_initial_input(data_dir, action_class):

	action_data, T = train.load_data(data_dir, action_class)
	lucky_ind = np.random.choice(len(action_data), 1) #randomly choose one instance! 
	print("lucky ind is ", lucky_ind)
	#reshape to 1*30*T
	stacked = np.vstack((action_data[lucky_ind]['pos_world'][0], action_data[lucky_ind]['pos_world'][1]))
	stacked = np.expand_dims(stacked, axis=0)
	return stacked


if __name__ == '__main__':
	initial_input = generate_initial_input('data/joint_positions', 'shoot_bow')
	pred_result = sample('data/', initial_input)
	animation.animate_action(pred_result, 'Prediction of shoot_bow')
