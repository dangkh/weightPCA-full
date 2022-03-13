# Interpolation missing marker in human motion
prerequisite:
	
	* python
	* opencv

algorithms folder contains the:

	- PCA_R1 and PCA_R2 re-implemented in Python (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152616)
	- our WPCA and WPCAG

Data and parameter before runing:
	
	arguments.py: this file will show what data will be used, change reference data, testing data, ... in this file.
		- data_link = "./data3D/HDM5.txt" is the location of testing data file)
		- length3D = 400 (length of testing sample)
		- missing_index = [[1050, 1450]] (frame index of testing sample in the data file)
		
		Additional dataset will include the data in both the testing sample and in the collection :

		- inner_testing = [[3050, 3450], [1860, 2260]] (frame index of resource extracted in testing sample for additional dataset)
		
		- collection_link = ["./data3D/HDM5.txt", "./data3D/135_01.txt", "./data3D/135_03.txt"] (indicates file locations for additional dataset)
		- outter_testing = [[650, 1050], [1450, 1850], [1050, 1450]] (frame index of data regarding to collection_link extracted for additional dataset)
		
		- test_link = "./test/" (indicate the location of the testing matrix. the testing matrix is 0/1 matrix while 0 refers the missing marker.)
		- save_dir = "C:/Users/nvmnghia/Desktop/weightPCA/list_result/" (indicate the wanted location for returned result)
		- default_test_link = "./test.txt" (indicate the test location in case of wrong test_link)

Run multiple test

	main.py: this file run multiple test in test_link
		
	
Genrerate missing matrix:
		
		- generate_mm_single.py (generate a single specific missing marker in the testing sample)
		- generate_mm_multiple.py (generate multiple missing markers in the testing sample)

visualization functions in plot_visual_files:
		
		detail will be updated.