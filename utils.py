from PIL import Image
import os        #引入os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

def name_gate(filePath):        
    file_name = list()        #新建列表
    for i in os.listdir(filePath):        #获取filePath路径下所有文件名
        data_collect = ''.join(i)        #文件名字符串格式
        file_name.append(data_collect)        #将文件名作为列表元素填入    
    return(file_name)        #返回列表

def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x

class MyDataset(Dataset):
    
    def __init__(self, txt_path, data_path):        
        fh = open(txt_path, 'r') #读取 制作好的txt文件的 图片路径和标签到imgs里
        imgs = []
        datapath = name_gate(data_path)
        print(datapath)
        for line in fh:
            line = line.rstrip()
            words = line.split()
            fn = str(data_path) +'/'+ str(datapath[int(words[1])]) + '/' +words[0]
            print(fn)
            img = Image.open(fn).convert('RGB')
            img = normalize(img)
            imgs.append((img, int(words[1])))
            self.imgs = imgs
    
    def __len__(self):
        return len(self.imgs)
        
