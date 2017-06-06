import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import check_mat


def animate_action(mat, title):
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
		plt.scatter(dots[0], dots[1], s=30, alpha=1)

	#animate fig
	scatAnim = animation.FuncAnimation(fig, update_dots, blit=False, frames=xrange(numframes), interval = 1, fargs=(mat,))
	plt.show()
	filename = title+'.mp4'
	scatAnim.save(filename, fps=15, extra_args=['-vcodec', 'libx264'])

def animate_two(mat1, mat2):
	
	plt.subplot(1,2,1)
	animate_action(mat1, "mat1")
	#=plt.subplot(1,2,2)
	animate_action(mat2, "mat2")

#debugging
if __name__ == '__main__':
	full_action_dict, actions_pruned = check_mat.save_actions('data/joint_positions')
	option1 = full_action_dict['shoot_bow'][actions_pruned['shoot_bow'][5]]['pos_world']
	#option2 = full_action_dict['shoot_bow'][actions_pruned['shoot_bow'][2]]['pos_world']
	
	#animate_two(option1, option2)

	animate_action(option1, "option1")



