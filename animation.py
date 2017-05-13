import matplotlib.pyplot as plt
import matplotlib.animation as animation
import check_mat

def animate_action(mat):
	print('shape of mat is ', mat.shape)

	numframes = mat.shape[2]
	numpoints = mat.shape[1]

	fig = plt.figure()
	scat = plt.scatter(mat[0,:,0], mat[1,:,0], s=100)

	def update_plot(i, var):
		print("i is ", i, " var is ", var)
		print("newPos is ", mat[:,:,i].shape)
		scat.set_offsets([mat[0,:,i],mat[1,:,i]])
		return scat

	ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes), interval = 50, fargs=(mat,))
	plt.show()

full_action_dict, actions_pruned = check_mat.save_actions('joint_positions')
animate_action(full_action_dict['walk'][0]['pos_world'])