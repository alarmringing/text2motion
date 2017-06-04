import tensorflow as tf
import numpy as np 
import train
import animation

def sample(model_path, initial_input):
	
	with tf.Session() as sess:
		#restores the model 
		saver = tf.train.import_meta_graph(model_path + 'model_lstm.meta')
		saver.restore(sess, model_path + 'model_lstm')
		#initialize session and graph
		sess.run(tf.global_variables_initializer())
		graph = tf.get_default_graph()
		pred_result = np.zeros((2, 15, len(initial_input)-1)) #this is fixed

		#at each timestep
		initial_timestep = initial_input[:,:,0:2]
		for t in range(initial_input.shape[2]-1):
			values = {
				"input_data:0": initial_timestep
			}
			predictions = sess.run([graph.get_tensor_by_name("regression/predictions:0")], values)
			print("shape of predictions[0] is ", predictions[0].shape)
			pred_result[t%2, t/2, t] = predictions[0]
			initial_timestep = np.vstack((predictions[0], np.zeros((30)))) #second value is meaningless
	
		return pred_result

def generate_initial_input(data_dir, action_class):

	action_data, T = train.load_data(data_dir, action_class)
	lucky_ind = np.random.choice(len(action_data), 1) #randomly choose one instance! 
	print("lucky ind is ", lucky_ind)
	#reshape to 1*30*T
	stacked = np.reshape(action_data[lucky_ind]['pos_world'], (1, -1, T)) 
	return stacked


if __name__ == '__main__':
	initial_input = generate_initial_input('data/joint_positions', 'shoot_bow')
	pred_result = sample('data/', initial_input)
	animation.animate_action(pred_result, 'Prediction of shoot_bow')
