import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import check_mat


def animate_action(mat):
	'''
	Takes an joint position matrix 2*15*framerate, and animates it on pyplot.
	'''
	
	numframes = mat.shape[2]
	numpoints = mat.shape[1]

	fig = plt.figure()
	plt.axis('scaled') #keep axis scaled
	plt.gca().invert_yaxis() #flip yaxis because it's negative. why? 
	scat = plt.scatter(mat[0,:,0], mat[1,:,0], s=100)
	plt.autoscale(False)
	lines = plt.plot([],[])

	#updates the scatter dots -- not showing right now, there's a bug
	def update_dots(i, mat):
		scat.set_offsets([mat[0,:,i],mat[1,:,i]])
		connect_plot(mat[:,:,i])
		return scat

	#connect between joints for readability
	def connect_plot(dots):
		connections = [(0,1), (0,2), (0,3), (0,4), (1,5), (1,6), (3,7), (4,8), (5,9), (6,10), \
		(7,11), (8,12), (9,13), (10, 14)]
		plt.cla()
		for pair in connections:
			ind = np.array(list(pair))
			plt.plot(dots[0, ind], dots[1, ind])

	#animate fig
	scatAnim = animation.FuncAnimation(fig, update_dots, blit=False, frames=xrange(numframes), interval = 1, fargs=(mat,))
	plt.show()

#debugging
full_action_dict, actions_pruned = check_mat.save_actions('joint_positions')
animate_action(full_action_dict['throw'][actions_pruned['throw'][5]]['pos_world'])

