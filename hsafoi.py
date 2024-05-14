import os
import math

txt_path = ['tarin_cvbrct_ground.txt', 'val_cvbrct_ground.txt','test_cvbrct_ground.txt',
            'tarin_cvbrct_air.txt', 'val_cvbrct_air.txt', 'test_cvbrct_air.txt',
            'tarin_airound_ground.txt', 'val_airound_ground.txt', 'test_airound_ground.txt',
            'tarin_airound_air.txt', 'val_airound_air.txt','test_airound_air.txt']

def generate(dir, label, idx):
	files = os.listdir(dir)
	files.sort()
	print(txt_path[idx*3])
	print(txt_path[idx*3+1])
    print(txt_path[idx*3+2])
	trainText = open(txt_path[idx*3],'a')	
	for file in files[:math.floor(len(files)*0.72)]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		trainText.write(name)
	trainText.close()

	valText = open(txt_path[idx*3+1],'a')	
	for file in files[math.floor(len(files)*0.72)+1 : math.floor(len(files)*0.8)]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		valText.write(name)
	valText.close()

	testText = open(txt_path[idx*3+1],'a')	
	for file in files[math.floor(len(files)*0.8)+1 : ]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		testText.write(name)
	testText.close()
 
 
outer_path = '/Public/WenbinHe/stage1/data'   #这里是你的图片的目录
 
 
if __name__ == '__main__':	
	folderlist0 = os.listdir(outer_path)          #列举文件夹
	idx = 0
	for branch1 in folderlist0:
		path1 = os.path.join(outer_path, branch1)
		folderlist1 = os.listdir(path1)          #列举文件夹

		for _, branch2 in enumerate(folderlist1):
			path2 = os.path.join(path1, branch2)
			folderlist2 = os.listdir(path2)    #列举文件夹
			folderlist2.sort()
			print(path2)
			for i, branch3 in enumerate(folderlist2):                   
				generate(os.path.join(path2, branch3), i, idx)
				print('*****' + branch3)
				i += 1
			idx += 1
			