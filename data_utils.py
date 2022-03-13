import numpy as np


def read_tracking_data(data_dir, ingore_confidence=False):
	Tracking2D = []
	f=open(data_dir, 'r')
	j=0
	for line in f:
		Tracking2D.append([])
		elements = line.split(',')
		Tracking2D[j].append( [elements[i] for i in range(len(elements))] )
		j+=1
	f.close()
	Tracking2D = np.array(Tracking2D) # list can not read by index while arr can be
	Tracking2D = np.squeeze(Tracking2D)

	if ingore_confidence:
		Tracking2D = Tracking2D.reshape(Tracking2D.shape[0],25,3)
		Tracking2D = Tracking2D[..., 0:2]
		Tracking2D = Tracking2D.reshape(Tracking2D.shape[0],50)
	return Tracking2D

def read_tracking_data3D(data_dir):
	Tracking3D = []
	f=open(data_dir, 'r')
	j=0
	for line in f:
		if j > 95:
			elements = line.split(' ')
			Tracking3D.append(list(map(float, elements[:-1])))
		j+=1
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	restore = np.copy(Tracking3D)
	Tracking3D = Tracking3D.reshape(Tracking3D.shape[0], 15, 6)
	Tracking3D = Tracking3D[..., 0:3]
	Tracking3D = Tracking3D.reshape(Tracking3D.shape[0],45)
	return Tracking3D, restore


def read_tracking_data3D_v2(data_dir):
	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line.split(',')
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	print(Tracking3D.shape)
	restore = np.copy(Tracking3D)
	return Tracking3D, restore

def read_tracking_data3D_v3(data_dir):
	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line.split(' ')
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	print(Tracking3D.shape)
	restore = np.copy(Tracking3D)
	return Tracking3D, restore

def read_tracking_data3D_nan(data_dir):
	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line[:-1].split(',')
		for x in range(len(elements)):
			if elements[x] == "NaN":
				elements[x] = '0'
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	print(Tracking3D.shape)
	restore = np.copy(Tracking3D)
	return Tracking3D, restore


def find_full_matrix(Tracking2D, frame_length, overlap=False):
	count = 0
	list_full = []
	for i, frame in enumerate(Tracking2D):
		if frame[frame==0].shape[0] == 0:
			count += 1
			if count >= frame_length:
				list_full.append([i - frame_length + 1, i + 1])
				if not overlap:
					count = 0
	return list_full

# Todo
def find_miss_matrix(Tracking2D, frame_length, bum_frame_miss, max_miss_in_a_frame):
	pass

# Todo
# plot miss_matrix on video to check
def contruct_sellect_matrix():
	pass



def reconstruct_3D(data_dir, new_dir,restore, predicted_matrix):
	restore = restore.reshape(restore.shape[0], 15, 6)
	predicted_matrix = predicted_matrix.reshape(predicted_matrix.shape[0], 15, 3)
	restore[..., 0:3] = predicted_matrix
	restore = restore.reshape(restore.shape[0],90)
	print("starting reconstruct")
	old_data = []
	f=open(data_dir, 'r')
	j=0
	for line in f:
		j+=1
		if j > 95:
			break
		old_data.append(line)
	f.close()
	f=open(new_dir, 'w')
	for line in old_data:
		f.write(line)
	for line in restore:
		tmp_str = ' '.join(str(e) for e in line)
		f.write(tmp_str+"\n")
	f.close()
	print("Finished reconstruct")