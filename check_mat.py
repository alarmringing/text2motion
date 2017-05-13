import scipy.io as sio
import collections
import os

def save_actions():
	'''
	crawls through the data and gathers info into a big dictionary, full_action_dict

	full_action_dict:
	key: action type
	value: list of mat files

	actions_oriented: 
	key: action type
	value: indices of mat files that follow the most popular viewpoint within this action
	'''
	full_action_dict = collections.defaultdict(list)
	actions_oriented = collections.defaultdict(list)
	data_dir = 'joint_positions'
	actions = []

	for action in os.listdir(data_dir):
		if not action.startswith('.'):
			actions.append(action)

	for action in actions: 
		actiondir = data_dir + "/" + action
		orientations = []

		#walks through each mat and saves them to full_action_dict
		for folder in os.listdir(actiondir): 
			if folder.startswith('.'): #skips weird .AppleDouble -- what does this mean? 
				continue 
			joints = sio.loadmat(actiondir + '/' + folder + '/joint_positions.mat')
			full_action_dict[action].append(joints)
			orientations.append(joints['viewpoint'][0])

		#find what's the most common viewpoint for this action class, and saves indices for 
		#mats with that orientation into actions_oriented
		max_viewpoint = max(set(orientations), key=orientations.count)
		actions_oriented[action] = [i for i,v in enumerate(full_action_dict[action]) if v['viewpoint'][0] == max_viewpoint]

	return full_action_dict, actions_oriented
