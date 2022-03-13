import sys
sys.path.append("/Users/kieudang/Desktop/weightPCA/")
import numpy as np
from data_utils import *
# from render_utils import *
# from algorithm import *
# from arguments import arg
# import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def read_data(link):
	matrix = []
	f=open(link, 'r')
	for line in f:
		elements = line.split(', ')
		matrix.append(list(map(float, elements)))
	f.close()

	matrix = np.array(matrix) 
	matrix = np.squeeze(matrix)
	return matrix

def read_new_bvh(link):
	matrix = []
	f=open(link, 'r')
	for line in f:
		elements = line.split(' ')
		matrix.append(list(map(float, elements)))
	f.close()

	matrix = np.array(matrix) 
	matrix = np.squeeze(matrix)
	deleted_joint = [5, 9, 14, 18]
	counter = 0
	new_matrix = np.zeros((matrix.shape[0], matrix.shape[1]+12))
	for xx in range(new_matrix.shape[1]):
		joint_order_number = xx // 3
		joint_order = xx % 3
		if joint_order_number in deleted_joint:
			if joint_order == 0:
				counter = counter-3
		new_matrix[:, xx] = matrix[:, counter]
		counter += 1
	print(new_matrix.shape)
	return new_matrix


if __name__ == '__main__':
	# 135 CMU
	dad_arr = [[0, 2], [1, 3], [2, 4], [3, 4], [9, 10], [10, 11], [11, 12], [12, 14], [14, 13], [14, 15], 
	[16, 17], [17, 18], [18, 19], [19, 21], [21, 20], [21, 22],
	[23, 27], [27, 28], [28, 29], [29, 30], [30, 31], [30, 32], [30, 33], 
	[24, 34], [34, 35], [35, 36], [36, 37], [37, 38], [37, 39], [37, 40],
	[4, 5],  [9, 4], [16, 4], [25, 23], [24, 26], [5, 25], [5, 26],[7, 23], [7, 24], [6, 16], [6, 9], [6, 7],
	[4, 8], [5, 8]
	]
	# [7, 23], [7, 24, [6, 16], [6, 9], [6, 7]]
	# Bvh
	# dad_arr = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8], [8, 9], [1, 10], 
	# [10, 11], [11, 12], [12, 13], [13, 14], [10, 15], [15, 16], [16, 17], [17, 18]]
	# 0 head
	# 1 neck
	# 2 LShoulder
	# 3 LElbow
	# 4 LWrist
	# 5
	# 6 RShoulder
	# 7 RElbow
	# 8 RWrist
	# 9
	# 10 Midhip
	# 11 LHip 
	# 12 LKnee
	# 13 LAnkle
	# 14 
	# 15 RHip
	# 16 RKnee
	# 17 RAnkle
	# 18
	# Tracking3D  = read_new_bvh("./render/1/original.txt")
	# Tracking3D  = read_new_bvh("./render/1/our_method.txt")
	# Tracking3D  = read_new_bvh("./render/1/PCA.txt")
	# Tracking3D  = read_data("./result.txt")
	# data_link = "./data3D/fastsong7.txt"
	prefix = "/Users/kieudang/Desktop/weightPCA/"
	data_link = prefix + "/data3D/135_02.txt"
	Tracking3D, _  = read_tracking_data3D_v2(data_link)
	Tracking3D = Tracking3D.astype(float)
	# np.savetxt("ChaiMue_take_001_Data.txt", Tracking3D, fmt = "%.3f", delimiter = ", ")
	fig = plt.figure()
	plt.style.use('fivethirtyeight')
	fig.set_size_inches(8, 12)
	ax = fig.add_subplot(111, projection='3d')
	fig.set_size_inches(16, 8)
	shift = 800
	shift2 = 800
	for index in range(2, Tracking3D.shape[0]):
		
		if index > 40: 
			break
		# plt.cla()
		print(index)
		frame = Tracking3D[50*index]
		n_joints = len(frame) // 3
		xs = []
		ys = []
		zs = []
		missing_point = 8
		for x in range(n_joints):
			if x != missing_point:
				xs.append(frame[x*3] + index * shift2)
				ys.append(frame[x*3+1] + index * shift)
				zs.append(frame[x*3+2])
			else:
				xs.append(index * shift2)
				ys.append(index * shift)
				zs.append(0)
		# print(xs)
		# print(ys)
		# print(zs)
		# ax.plot(xs, ys, zs, 'r.')
		# tmp = 1
		# for x in range(len(dad)):
		# 	xxs = [xs[tmp],xs[tmp+1]]
		# 	yys = [ys[tmp],ys[tmp+1]]
		# 	zzs = [zs[tmp],zs[tmp+1]]
		# 	ax.plot(xxs, yys, zzs, 'b')
		# xs[missing_point] = 0
		# ys[missing_point] = 0
		# zs[missing_point] = 0

		for x in range(len(dad_arr)):
			dad = dad_arr[x][0]
			child = dad_arr[x][1]
			my_color = 'b'
			if dad != missing_point and child != missing_point:
				# if (dad == missing_point) or (child == missing_point):
				# 	my_color = 'g'
				# print(xs[dad],xs[child])
				xxs = [xs[dad],xs[child]]
				yys = [ys[dad],ys[child]]
				zzs = [zs[dad],zs[child]]
				ax.plot(xxs, yys, zzs, my_color)

		# xs = np.asarray(xs)
		# ys = np.asarray(ys)
		# zs = np.asarray(zs)
		for idx in range(n_joints):
			if idx != 8:
				ax.plot([xs[idx]], [ys[idx]], [zs[idx]], 'r.')
		# CMU
		# ax.set_xlim(-1000, 1000)
		# ax.set_ylim(-500, 2500)
		# ax.set_zlim3d(0, 1500)

		# BVH
		# ax.set_xlim3d(-70, 70)
		# ax.set_ylim3d(-50, 300)
		# ax.set_zlim3d(0, 300)


		# ax.set_xlabel('X Label')
		# ax.set_ylabel('Y Label')
		# ax.set_zlabel('Z Label')
		ax.view_init(elev=4., azim=-1)
		# plt.savefig("filename"+ str(index)+".png")
		plt.yticks(visible=False)
		plt.xticks(visible=False)
		# plt.savefig('filename'+ str(index) +'.eps', format='eps')
		plt.pause(0.1)
		# tmp_img = plt.gcf()

	plt.show()
	plt.close()
	print("done")
