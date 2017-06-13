import tensorflow as tf
import numpy as np 
import train
import sys
import animation
import sample

#Given two predictions, 
def combine_sample(pred_one, pred_two, replaced_joints):
	pred_res = np.copy(pred_one)
	for ind in replaced_joints: 
		pred_res[:,ind,:] = pred_two[:,ind,:]
	return pred_res

if __name__ == '__main__':

	model_one = sys.argv[1]
	action_one = sys.argv[2]
	model_two = sys.argv[3]
	action_two = sys.argv[4]
	test_ind = sys.argv[5]
	prune_viewpoints = False
	pre_step = 10

	top_body = [0,2,3,4,7,8,11,12]

	#pull out initial input
	seed_action = sample.extract_action('data/joint_positions', action_one, prune_viewpoints, test_ind)
	initial_input = sample.generate_initial_input(seed_action)


	pred_one = sample.sample('data/', initial_input, model_one, pre_step)
	pred_two = sample.sample('data/', initial_input, model_two, pre_step)

	pred_res = combine_sample(pred_one, pred_two, top_body)

	anim_name = action_one + "_and_" + action_two
	animation.animate_action(pred_res, anim_name, test_ind,'Prediction of %s'%anim_name)