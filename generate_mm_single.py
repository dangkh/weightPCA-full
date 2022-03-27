# generate missing gap in specific joint according to PLOS16
import numpy as np
from data_utils import *
from arguments import arg
import sys
import random
from datetime import datetime
import os
def generate_missing_joint_gap1(n, m, marker = None):
	# CMU: 
	LSHO = 9
	LKNE = 28 
	LWA = 14
	# HDM: 
	# LSHO = 18
	# LKNE = 3 
	# LWA = 21

	frames = 385
	matrix = np.ones((n,m))
	joints = np.arange(m//3)
	counter = 0
	total_numgap = 1
	start_missing_frame = random.randint(0, max(0,n-frames))
	print('start : ')
	while counter < total_numgap:
		print("start_missing_frame: ",start_missing_frame, ";")
		# missing_joint = np.random.randint(0,m//3)
		if marker is None:
			missing_joint = LWA
		else:
			missing_joint = marker
		for frame in range(start_missing_frame, start_missing_frame+frames):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
		old_start = start_missing_frame
		start_missing_frame = random.randint(old_start, min(n-frames, old_start+frames))
		counter+=1
	print("end.")
	counter = 0
	for x in range(n):
		for y in range(m):
			if matrix[x][y] == 0: counter +=1
	print("percent missing: ", 100.0 * counter / (n*m))
	return matrix

def generate_missing_joint_gap(n, m):
	# # CMU
	# LSHO = 9
	# LKNE = 28 
	# LWA = 14
	# HDM: 
	LSHO = 18
	LKNE = 3 
	LWA = 21

	frames = 300
	matrix = np.ones((n,m))
	counter = 0
	total_numgap = 3
	markersp = [-1,0,1]
	start_missing_frame = random.randint(1, n-frames-50)
	print('start : ')
	while counter < total_numgap:
		print(counter, ": ",start_missing_frame, ";")
		missing_joint = LSHO + markersp[counter]
		for frame in range(start_missing_frame, start_missing_frame+frames):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
		old_start = start_missing_frame
		start_missing_frame = random.randint(old_start, min(n-frames, old_start+frames))
		counter+=1
	print("end.")
	counter = 0
	for x in range(n):
		for y in range(m):
			if matrix[x][y] == 0: counter +=1
	print("percent missing: ", 100 * counter / (n*m))
	return matrix



def process_gap_missing():
	
	test_location = arg.test_link
	gaps = [1]
	test_reference = arg.missing_index
	sample = np.copy(Tracking3D[test_reference[0]:test_reference[1]])

	for times in range(50):
		print("current: ", times)
		
		missing_matrix = generate_missing_joint_gap1(sample.shape[0], sample.shape[1])		

			
		np.savetxt(test_location+"/"+str(gaps[0])+"/"+str(times)+ ".txt", missing_matrix, fmt = "%d")
	# f = open(test_location + "/info.txt", "w")
	# f.write(str(datetime.now()))
	# f.close()		
	return 

def process_gap_missing_all(marker):
	
	test_location = arg.test_link
	if os.path.isdir(test_location + str(marker)):
		print("ok", test_location + str(marker))
	else:
		print("not ok")
		os.makedirs(test_location + str(marker))
	gaps = [1]
	test_reference = arg.missing_index
	sample = np.copy(Tracking3D[test_reference[0]:test_reference[1]])

	for times in range(50):
		print("current: ", times)
		
		missing_matrix = generate_missing_joint_gap1(sample.shape[0], sample.shape[1], marker = marker)		

			
		np.savetxt(test_location+str(marker)+"/"+str(times)+ ".txt", missing_matrix, fmt = "%d")
	# f = open(test_location + "/info.txt", "w")
	# f.write(str(datetime.now()))
	# f.close()		
	return 

if __name__ == '__main__':

	Tracking3D, _  = read_tracking_data3D_v2(arg.data_link)
	Tracking3D = Tracking3D.astype(float)
	for i in range(41):
		process_gap_missing_all(i)
	# process_gap_missing()
