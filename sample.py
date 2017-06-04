import tensorflow as tf
import numpy as np 
import train
import animation

def sample(model_path, initial_input):
	print("running sample! trying to restore the model")
	#restores the model 
	saver = tf.train.import_meta_graph(model_path + 'model.meta')
	saver.restore(sess, model_path + "model.ckpt")
	print("Maybe model was restored. I'm not sure")
	sess.run(tf.global_variables_initializer())

	values = {
		input_data: initial_input
	}
	predictions = sess.run([self.pred], initial_input)
	print("predictions.shapei s ", predictions.shape)

	pred_result = np.zeros((2, 15, len(pred_result))) #this is fixed
	for i in range(len(predictions)): #for each timestep of prediction
		pred_result[i%2, i/2, i] = pred_result[i]

	return pred_result

def generate_initial_input(data_dir, action_class):

	action_data, T = train.load_data(data_dir, action_class)
	lucky_ind = np.random.choice(len(action_data), 1) #randomly choose one instance! 
	print("lucky ind is ", lucky_ind)
	#reshape to 1*30*T
	stacked = np.reshape(action_data[lucky_ind]['pos_world'], (1, -1, T)) 
	#return first timestep of this ,1*30*1
	return stacked[:,:,0]



if __name__ == '__main__':
	initial_input = generate_initial_input('data/joint_positions', 'shoot_bow')
	pred_result = sample('data/', initial_input)
	animation.animate_action(pred_result, 'Prediction of shoot_bow')
