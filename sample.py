import tensorflow as tf
import numpy as np 
import train
import sys
import animation

def sample(model_path, initial_input, savefile_name, pre_step):
	
	def mean_squared_error(pred, truth):
		return np.mean((pred - truth)**2)/(30)

	with tf.Session() as sess:
		#restores the model 
		saver = tf.train.import_meta_graph(model_path + savefile_name +'.meta')
		saver.restore(sess, model_path + savefile_name)
		#initialize session and graph
		sess.run(tf.global_variables_initializer())
		graph = tf.get_default_graph()
		T = initial_input.shape[2]
		pred_result = np.zeros((2, 15, T)) #this is fixed

		#initial timesteps
		pred_result[0, :, :pre_step] = initial_input[0,:15,:pre_step]
		pred_result[1, :, :pre_step] = initial_input[0,15:,:pre_step]

		#INPUT SHOULD BE IN BATCH * 30 * TIMESTEP
		#initial input

		initial_timestep = np.reshape(initial_input[0,:,:pre_step], (1, -1, pre_step))
		timestep_input = \
			np.concatenate((initial_timestep, np.zeros((1,30,T-pre_step))), axis=2)
		#at each timestep
		for t in range(pre_step-1,initial_input.shape[2]-1):
			values = {
				"input_data:0": timestep_input
			}
			predictions = sess.run([graph.get_tensor_by_name("regression/predictions:0")], values)
			#PREDICTIONS WILL BE IN T * BATCH * 30
			this_pred = predictions[0][t, 0, :]
			pred_result[0, :, t+1] = this_pred[:15]
			pred_result[1, :, t+1] = this_pred[15:]
			timestep_input = np.concatenate((timestep_input[:,:,:t], \
			np.reshape(this_pred, (1,-1,1)), np.zeros((1,30,T-t-1))), axis=2)
			
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

	savefile_name = sys.argv[1]
	pre_step = 10
	initial_input = generate_initial_input('data/joint_positions', 'shoot_bow')
	pred_result = sample('data/', initial_input, savefile_name, 10)
	#animation.animate_action(np.reshape(initial_input, (2, 15, 40)), 'original shoot_bow')
	animation.animate_action(pred_result, 'Prediction of shoot_bow')
