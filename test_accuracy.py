import tensorflow as tf  
import numpy as np 
import sample 
import sys
import animation

def report_test_error(savefile_name, action_class, test_inds, pre_step, prune_viewpoints):
	summ = 0
	for ind in test_inds:
		ind = int(ind)
		print("Checking error for test_ind ", ind)
		action = sample.extract_action('data/joint_positions', action_class, ind, prune_viewpoints)
		initial_input = sample.generate_initial_input(action)
		pred_result = sample.sample('data/', initial_input, savefile_name, pre_step)
		animation.animate_action(pred_result, action_class, ind,'Prediction of %s'%action_class)
		summ += sample.mean_squared_error(pred_result, action)
	return summ / len(test_inds)

if __name__ == '__main__':

	savefile_name = sys.argv[1]
	action_class = sys.argv[2]
	test_inds = sys.argv[3]
	pre_step = 10
	prune_viewpoints = False
	err = report_test_error(savefile_name, action_class, test_inds.split(','), pre_step, prune_viewpoints)
	print("Average test error for " + action_class + " is ", err)
