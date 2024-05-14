import os
import math

txt_path = ['tarin_airound_air.txt', 'test_airound_air.txt', 
			   'tarin_airound_ground.txt', 'test_airound_ground.txt',
				'tarin_cvbrct_air.txt', 'test_cvbrct_air.txt', 
			  	'tarin_cvbrct_ground.txt', 'test_cvbrct_ground.txt']

def generate(dir, label, idx):
	files = os.listdir(dir)
	files.sort()
	print(txt_path[idx*2])
	print(txt_path[idx*2+1])
	trainText = open(txt_path[idx*2],'a')	
	for file in files[:math.floor(len(files)*0.9)]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		trainText.write(name)
	trainText.close()

	trainText = open(txt_path[idx*2+1],'a')	
	for file in files[math.floor(len(files)*0.9)+1 : ]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		trainText.write(name)
	trainText.close()
 
 
outer_path = 'E:\\Document\\stage1\\data'   #这里是你的图片的目录
 
 
if __name__ == '__main__':	
	folderlist0 = os.listdir(outer_path)          #列举文件夹
	idx = 0
	for branch1 in folderlist0:
		path1 = os.path.join(outer_path, branch1)
		folderlist1 = os.listdir(path1)          #列举文件夹

		for _, branch2 in enumerate(folderlist1):
			path2 = os.path.join(path1, branch2)
			folderlist2 = os.listdir(path2)    #列举文件夹
			print(path2)
			for i, branch3 in enumerate(folderlist2):                   
				generate(os.path.join(path2, branch3), i, idx)
				print('*****' + branch3)
				i += 1
			idx += 1
			