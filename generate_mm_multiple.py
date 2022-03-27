# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from arguments import arg
import sys
import random
from datetime import datetime

def generate_missing_joint_gap(n, m, number_gap):
	frames = 350
	matrix = np.ones((n,m))
	joints = np.arange(m//3 - 1)
	np.random.shuffle(joints)
	# missing_joint = 14
	counter = 0
	while counter < number_gap:
		missing_joint = joints[counter]
		counter+=1
		start_missing_frame = random.randint(1, n-frames)
		print("start_missing_frame: ", start_missing_frame, "joint: ", missing_joint)
		for frame in range(start_missing_frame, start_missing_frame+frames):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
		missing_joint = joints[int(counter / 2)] + (counter % 2)
		# if counter % 2 == 0:
		# 	start_missing_frame = random.randint(1, n-frames-50)
		# old_start = start_missing_frame

		# start_missing_frame = random.randint(old_start, min(n-frames, old_start+frames/2+1))
	counter = 0

	for x in range(n):
		for y in range(m):
			if matrix[x][y] == 0: counter +=1
	print("percent missing: ", 100.0 * counter / (n*m))
	return matrix


def process_gap_missing():
	test_location = arg.test_link
	gaps = [3, 6, 9]
	test_reference = arg.missing_index
	sample = np.copy(Tracking3D[test_reference[0]:test_reference[1]])

	for gap in gaps:
		for times in range(200):
			print("current: ", gap, times)
			missing_matrix = generate_missing_joint_gap(sample.shape[0], sample.shape[1], gap)
			np.savetxt(test_location+ str(gap) +"/"+str(times)+ ".txt", missing_matrix, fmt = "%d")
	# f = open("./test_data_CMU_gap/info.txt", "w")
	# f.write(str(datetime.now()))
	# f.close()		
	return 


if __name__ == '__main__':

	Tracking3D, _  = read_tracking_data3D_v2(arg.data_link)
	Tracking3D = Tracking3D.astype(float)
	process_gap_missing()
