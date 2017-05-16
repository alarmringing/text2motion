import scipy.io as sio
import collections
import os
import copy
import check_mat

data_dir = 'joint_positions'
full_action_dict, actions_pruned = check_mat.save_actions('joint_positions')


#actions_pruned['walk']
"""
A mask of indices 
to reduce 15 nodes on the skeleton into 4 clusters for the factor graph
"""
nodeFeaturesRanges={}
nodeFeaturesRanges['torso'] = range(6)
nodeFeaturesRanges['right_arm'] = [6,7,14,15,22,23]
nodeFeaturesRanges['left_arm'] = [8, 9, 16, 17, 24, 25]
nodeFeaturesRanges['right_leg'] = [10, 11, 18, 19, 26, 27]
nodeFeaturesRanges['left_leg'] = [12, 13, 20, 21, 24, 25, 28, 29] 




"""
action_data: (2, 15, nframes) array of normalized x,y positions for nframes of a video for given action in the dataset.

features: dictionary keyed by any of the nodes types defined on the skeleton
e-g, Features['torso']  an array of shape (6, nframes) representing a cluster of neck, belly, face points
"""

def clusterNodes(action_data):
    num_frames = action_data.shape[2]
    num_features = 15 
    data = action_data.reshape(num_features*2,num_frames)

    features = {}
    nodeNames = nodeFeaturesRanges.keys()
    for nm in nodeNames:
        filterList = nodeFeaturesRanges[nm]
        features[nm] = data[filterList,:]
    return features


#test
action_data = full_action_dict['walk'][0]['pos_world']
print clusterNodes(action_data)['torso']




"""
Future work:
when clustering nodes, ignore dims with small std
*add noise to features
*add noise to data

"""
