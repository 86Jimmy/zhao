import os
import math

def generate(dir, label):
	files = os.listdir(dir)
	files.sort()
	print('****************')
	print('input :',dir)
	print('start...')
	trainText = open('train_list.txt','a')	
	for file in files[:math.floor(len(files)*0.9)]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		trainText.write(name)
	trainText.close()
	trainText = open('test_list.txt','a')	
	for file in files[math.floor(len(files)*0.9)+1 : ]:
		fileType = os.path.split(file)
		if fileType[1] == '.txt':
			continue
		name = dir + '/' + file + ' ' + str(int(label)) +'\n'
		trainText.write(name)
	trainText.close()
 
 
outer_path = 'E:/Document/stage1/data/airound/aerial'   #这里是你的图片的目录
 
 
if __name__ == '__main__':	
	folderlist = os.listdir(outer_path)          #列举文件夹	 
	for idx, folder in enumerate(folderlist):		
			generate(os.path.join(outer_path, folder), idx)
			idx += 1			