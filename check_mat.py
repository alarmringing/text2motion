import scipy.io as sio
import collections
import os
import numpy as np
import random
from six.moves import cPickle

def fetch_action_data(action_label, full_action_dict, actions_pruned):
	return [full_action_dict[action_label][i] for i in actions_pruned[action_label]]

def save_actions(data_dir):
	'''
	crawls through the data and gathers info into a big dictionary, full_action_dict

	full_action_dict:
	key: action type
	value: list of mat files

	actions_pruned: 
	key: action type
	value: indices of mat files that follow the most popular viewpoint and frame# within this action
	'''
	full_action_dict = collections.defaultdict(list)
	actions_pruned = collections.defaultdict(list)
	actions = []

	for action in os.listdir(data_dir):
		if not action.startswith('.'):
			actions.append(action)

	for action in actions: 
		actiondir = data_dir + "/" + action
		orientations = []
		frames = []

		#walks through each mat and saves them to full_action_dict
		for folder in os.listdir(actiondir): 
			if folder.startswith('.'): #skips weird .AppleDouble -- what does this mean? 
				continue 
			joints = sio.loadmat(actiondir + '/' + folder + '/joint_positions.mat')
			full_action_dict[action].append(joints)
			#save orientation and frame info
			orientations.append(joints['viewpoint'][0])
			frames.append(len(joints['scale'][0]))

		#find what's the most common viewpoint & frame for this action class, and saves indices for 
		# mats with that orientation and frame # into actions_pruned
		max_viewpoint = max(set(orientations), key=orientations.count)
		max_frames = max(set(frames), key=frames.count)
		actions_pruned[action] = [i for i,v in enumerate(full_action_dict[action]) \
		if v['viewpoint'][0] == max_viewpoint and len(v['scale'][0]) == max_frames ]

	return full_action_dict, actions_pruned



'''
Loads the data in various ways for the tensorflow model.
If you don't save, 
only function used by 
'''
class DataLoader():
    def __init__(self, action_type, dim, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.dimension = dim
        self.num_batches = 1 #for now 

        #just declare paths
        input_file = os.path.join(data_dir, "joint_positions")
        pos_dict_file = os.path.join(data_dir, "pos_dict.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # read full action dict, actions pruned
    	full_action_dict, actions_pruned = save_actions(input_file)
    	self.data = fetch_action_data(action_type, full_action_dict, actions_pruned)

        if not (os.path.exists(pos_dict_file) and os.path.exists(tensor_file)):
            print("reading joint info")
            self.preprocess(input_file, action_type, pos_dict_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(pos_dict_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, pos_dict_file, tensor_file):
    	
    	# flattened form of different motion instances, JUST TO CALCULATE the bottom values
    	data = list()
    	for mat in self.data:
    		data += mat['pos_world'][self.dimension%2,self.dimension//2,:].tolist()
    	'''
    	IMPORTANT! 
    	FOR THE MILESTONE 
    	We're only looking at dimension 0 here.
    	'''
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars = np.arange(min(data), max(data), 0.1).tolist()
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(pos_dict_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)


    def create_batches(self):

    	'''
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        '''
        '''
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        #ydata is time-shifted to the right 1 step
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        '''

    def next_batch(self):
        #x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        #self.pointer += 1

        x = np.zeros((self.batch_size, self.seq_length))
        y = np.zeros((self.batch_size, self.seq_length))
        random.shuffle(self.data)
        files_to_use = self.data[:self.batch_size]
    	for i in range(len(files_to_use)):
    		mat = files_to_use[i]
    		x[i,:] = mat['pos_world'][self.dimension%2,self.dimension//2,:]
    		y[i,:-1] = x[i,1:]
    		y[i, -1] = x[i, 0]

        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0




