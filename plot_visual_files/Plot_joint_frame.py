# implemention corresponds 1st formula to check result
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec

# def draw(ax1, ax2, ax3, link_data, joint_index, list_frame, color_set, axins1, axins2, axins3):
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
	# axins1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	ax2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	# axins2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	ax3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	# axins3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	# return ax1, ax2, ax3, axins1, axins2, axins3
	return ax1, ax2, ax3


if __name__ == '__main__':
	color = ['g', 'r', 'y', 'b', 'k' , 'c']

	
	data_link = ["./missing_matrix.txt"]
	Tracking3D, _  = read_tracking_data3D_v3(data_link[0])
	Tracking3D = Tracking3D.astype(float)
	tmp = np.where(Tracking3D == 0)[0]
	list_frame = np.unique(tmp)
	ox = [x for x in range(Tracking3D.shape[0])]
	Tracking3D = Tracking3D.T



	data_link = ["./original.txt"]
	Tracking3D, _  = read_tracking_data3D_v3(data_link[0])
	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D.T
	number_joint = Tracking3D.shape[0] // 3
	
	fig = plt.figure()
	# fig.suptitle(title, fontsize=16)
	joint_index = 10
	x1 = joint_index*3
	x2 = joint_index*3+1
	x3 = joint_index*3+2
	xx = Tracking3D[x1]
	yy = Tracking3D[x2]
	zz = Tracking3D[x3]
	
	# gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1]) 
	ax1 = plt.subplot(311)
	ax1.set_ylabel(' x axis ')
	ax1.plot(ox[:list_frame[0]+1], xx[:list_frame[0]+1], color='black', linewidth=2)
	ax1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	ax1.plot(ox[list_frame[-1]-1:], xx[list_frame[-1]-1:], color='black', linewidth=2)
	# axins1 = zoomed_inset_axes(ax1, 2.5, loc=10,  bbox_to_anchor=(900,500))
	# axins1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	# x1, x2, y1, y2 =  20, 45, -50, -37
	# axins1.set_xlim(x1, x2) # apply the x-limits
	# axins1.set_ylim(y1, y2) # apply the y-limits
	# plt.yticks(visible=False)
	# plt.xticks(visible=False)
	# mark_inset(ax1, axins1, loc1=1, loc2=3, fc="none", ec="0.5")


	ax2 = plt.subplot(312)
	ax2.set_ylabel(' y axis ')
	ax2.plot(ox[:list_frame[0]+1], yy[:list_frame[0]+1], color='black', linewidth=2)
	ax2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	ax2.plot(ox[list_frame[-1]-1:], yy[list_frame[-1]-1:], color='black', linewidth=2)

	# axins2 = zoomed_inset_axes(ax2, 2.5, loc=10,  bbox_to_anchor=(900, 320))
	# axins2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	# x1, x2, y1, y2 =  20, 45, 108, 111
	# axins2.set_xlim(x1, x2) # apply the x-limits
	# axins2.set_ylim(y1, y2) # apply the y-limits
	# plt.yticks(visible=False)
	# plt.xticks(visible=False)
	# mark_inset(ax2, axins2 , loc1=1, loc2=3, fc="none", ec="0.5")

	ax3 = plt.subplot(313)
	ax3.set_ylabel(' z axis ')
	ax3.plot(ox[:list_frame[0]+1], zz[:list_frame[0]+1], color='black', linewidth=2)
	ax3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	ax3.plot(ox[list_frame[-1]-1:], zz[list_frame[-1]-1:], color='black', linewidth=2)

	# axins3 = zoomed_inset_axes(ax3, 2.5, loc=10,  bbox_to_anchor=(900, 160))
	# axins3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], linestyle="--",color='black', linewidth=1)
	# x1, x2, y1, y2 =  20, 45, 104, 109
	# axins3.set_xlim(x1, x2) # apply the x-limits
	# axins3.set_ylim(y1, y2) # apply the y-limits
	# plt.yticks(visible=False)
	# plt.xticks(visible=False)
	# mark_inset(ax3, axins3 , loc1=1, loc2=3, fc="none", ec="0.5")

	ax3.set_xlabel('Frames')

	# done original

	data_link = "./interpolate.txt"
	# ax1, ax2, ax3, axins1, axins2, axins3 = draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "b", axins1, axins2, axins3)
	ax1, ax2, ax3 = draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "b")
	
	data_link = "./PCA.txt"
	# draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "r", axins1, axins2, axins3)
	draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "r")
	
	fig.set_size_inches(16, 8)
	
	plt.savefig('filename.eps', format='eps')
	# plt.savefig('result.png')