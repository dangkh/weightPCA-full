# implemention corresponds 1st formula to check result
import numpy as np
import sys
sys.path.append("C:\\Users\\nvmnghia\\Desktop\\weightPCA")
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec

def draw(ax1, ax2, ax3, link_data, joint_index, list_frame, color_set):
	Tracking3D, _  = read_tracking_data3D_v3(link_data)
	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D.T
	ox = [x for x in range(Tracking3D.T.shape[0])]

	x1 = joint_index*3
	x2 = joint_index*3+1
	x3 = joint_index*3+2
	xx = Tracking3D[x1]
	yy = Tracking3D[x2]
	zz = Tracking3D[x3]
	ax1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	ax2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	ax3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	# axins3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	return ax1, ax2, ax3


if __name__ == '__main__':
	color = ['g', 'r', 'y', 'b', 'k' , 'c']
	prefix = "C:/Users/nvmnghia/Desktop/weightPCA/list_result"
	
	data_link = ["/missing_matrix.txt"]
	Tracking3D, _  = read_tracking_data3D_v3(prefix+data_link[0])
	Tracking3D = Tracking3D.astype(float)
	tmp = np.where(Tracking3D == 0)[0]
	list_frame = np.unique(tmp)
	joint_index = int(np.where(Tracking3D == 0)[1][0] / 3)
	ox = [x for x in range(Tracking3D.shape[0])]
	Tracking3D = Tracking3D.T

	data_link = ["/original.txt"]
	Tracking3D, _  = read_tracking_data3D_v3(prefix+data_link[0])
	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D.T
	number_joint = Tracking3D.shape[0] // 3
	
	fig = plt.figure()
	spec = fig.add_gridspec(ncols=1, nrows=3)
	# fig.suptitle(title, fontsize=16)
	x1 = joint_index*3
	x2 = joint_index*3+1
	x3 = joint_index*3+2
	xx = Tracking3D[x1]
	yy = Tracking3D[x2]
	zz = Tracking3D[x3]
	
	# gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1]) 
	rangeX = 400
	ax1 = fig.add_subplot(spec[0, 0])
	# ax1 = plt.subplot(321)
	ax1.set_ylabel(' x coordinate ')
	ax1.yaxis.set_label_coords(-0.05, 0.5)
	ax1.plot(ox[:list_frame[0]+1], xx[:list_frame[0]+1], color='black', linewidth=2)
	ax1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	ax1.plot(ox[list_frame[-1]-1:], xx[list_frame[-1]-1:], color='black', linewidth=2)

	# ax2 = plt.subplot(323)
	ax2 = fig.add_subplot(spec[1, 0])
	ax2.set_ylabel(' y coordinate ')
	ax2.yaxis.set_label_coords(-0.05, 0.5)
	ax2.plot(ox[:list_frame[0]+1], yy[:list_frame[0]+1], color='black', linewidth=2)
	ax2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	ax2.plot(ox[list_frame[-1]-1:], yy[list_frame[-1]-1:], color='black', linewidth=2)


	# ax3 = plt.subplot(325)
	ax3 = fig.add_subplot(spec[2, 0])
	ax3.set_ylabel(' z coordinate ')
	ax3.yaxis.set_label_coords(-0.05, 0.5)
	ax3.plot(ox[:list_frame[0]+1], zz[:list_frame[0]+1], color='black', linewidth=2)
	ax3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	ax3.plot(ox[list_frame[-1]-1:], zz[list_frame[-1]-1:], color='black', linewidth=2)

	ax3.set_xlabel('Frames')

	# done original

	
	data_link = "/WPCAG.txt"
	ax1, ax2, ax3 = draw(ax1, ax2, ax3, prefix+data_link, joint_index, list_frame, "g")
	# ax1, ax2, ax3 = draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "green", linestyle = '-.')
	

	data_link = "/WPCA.txt"
	draw(ax1, ax2, ax3, prefix+data_link, joint_index, list_frame, "r")
	# draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "r")
	
	fig.set_size_inches(16, 10)
	# plt.show()
	plt.savefig(prefix+'/CMU_37.eps', format='eps')
	# plt.savefig('result.png')