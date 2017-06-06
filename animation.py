import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.axes as axes
import numpy as np
import check_mat
import os


def animate_action(mat, action, test_ind, title):
	'''
	Takes an joint position matrix 2*15*framerate, and animates it on pyplot.

	Saves animation into an .mp4 file named title
	'''
	print("animating " + title)
	numframes = mat.shape[2]
	numpoints = mat.shape[1]
	fig = plt.figure()
	plt.title(title)
	plt.axis('scaled') #keep axis scaled

	plt.gca().invert_yaxis() #flip yaxis because it's negative. why? 
	scat = plt.scatter(mat[0,:,0], mat[1,:,0], s=100)
	plt.autoscale(False)


	lines = plt.plot([],[])

	#get_tight_layout()

	#updates the scatter dots -- not showing right now, there's a bug

	dir_name = os.path.join(os.getcwd(), action)

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	ind_dir = os.path.join(dir_name, '%s' %test_ind )

	if not os.path.exists(ind_dir):
		os.mkdir(ind_dir)

	def update_dots(i, mat):
		scat.set_offsets([mat[0,:,i],mat[1,:,i]])
		connect_plot(mat[:,:,i])

		if i % 5 == 0:

			img_title = os.path.join(ind_dir, 'sample_%d'%i)
			plt.savefig(img_title)

		return scat

	#connect between joints for readability
	def connect_plot(dots):
		connections = [(0,1), (0,2), (0,3), (0,4), (1,5), (1,6), (3,7), (4,8), (5,9), (6,10), \
		(7,11), (8,12), (9,13), (10, 14)]
		plt.cla()
		for pair in connections:
			ind = np.array(list(pair))
			plt.plot(dots[0, ind], dots[1, ind])
		plt.scatter(dots[0], dots[1], s=30, alpha=1)

	#animate fig
	scatAnim = animation.FuncAnimation(fig, update_dots, blit=False, frames=numframes, interval = 1, repeat=True, fargs=(mat,))
	
	#save as a file with frame rate of choice
	filename = action + "_" + "%s"%test_ind +'.mp4'
	filedir = os.path.join(dir_name, filename)
	scatAnim.save(filename, fps=15, extra_args=['-vcodec', 'libx264'])

	#show plt
	plt.show()

def animate_two(mat1, mat2):
	
	plt.subplot(1,2,1)
	animate_action(mat1, "mat1")
	#=plt.subplot(1,2,2)
	animate_action(mat2, "mat2")


#debugging
if __name__ == '__main__':
	full_action_dict, actions_pruned = check_mat.save_actions('data/joint_positions')
	#option1 = full_action_dict['shoot_bow'][actions_pruned['shoot_bow'][5]]['pos_world']

	#test
	title = 'wave'
	action = full_action_dict['wave'][actions_pruned['wave'][5]]['pos_world']

	animate_action(action, 'wave', 5, title)








