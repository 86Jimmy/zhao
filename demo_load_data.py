from PIL import Image
from scipy.io import savemat
import torch
import os        #引入os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import MVGG
from torch.autograd import Variable

def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x

class MyDataset(Dataset):
    
    def __init__(self, txt_path, transform = None, target_transform = None):        
        fh = open(txt_path, 'r') #读取 制作好的txt文件的 图片路径和标签到imgs里
        imgs = []                
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))                   
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]

        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img) 
        return img, label            
    
    def __len__(self):
        return len(self.imgs)
        
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),    
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )])

batch_size = 64
target, feature = [], []
train_data_list = MyDataset(txt_path='test_list.txt', transform=transform_train)
# for i in range(train_data_list):
#     data[i], label[i] = train_data_list[i]


train_loader  = DataLoader(train_data_list, batch_size=batch_size)

my_model = MVGG()

my_model.cuda()
print(my_model)

my_model.eval()

with torch.no_grad():
    for batch_idx, (data, label) in enumerate(train_loader):
        # data, label = data.transpose(0,1).cuda(), label.cuda()
        data = data.cuda()
        feature.extend(my_model(data).cpu())
        target.extend(label)      
    savemat('train.mat', {'train':feature, 'label': target})    
    print('ok')