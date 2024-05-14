from PIL import Image
from scipy.io import savemat
import torch
import torch.nn as nn 
from torchvision import models
from torch.utils.data import Dataset
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from dataset import MyDataset

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 100]')    
    parser.add_argument('--data_set', type=str, default='cvbrct', metavar='N',
                        help='which data_set would you like to backbone :[cvbrct, airound]') 
    opt = parser.parse_args(args=[])

    return opt

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

def mymodel():
    vgg = models.vgg11(pretrained = True)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-3])
    my_model = vgg
    
    print(my_model)
    return my_model    
    
def data_process(model, train_loader):    
    target, feature = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(train_loader):            
            data = data.cuda()
            feature.extend(model(data).cpu())
            target.extend(label)  
    print('ok\n')
    return feature, target
               
def main():
    opt = parse_option()
    
    # model
    model = mymodel()
    model.cuda()

    # data_load
    if(opt.data_set == "cvbrct"):
        ground_train = MyDataset(txt_path='tarin_cvbrct_street.txt', transform=transform_train)
        ground_val = MyDataset(txt_path='val_cvbrct_street.txt', transform=transform_train)
        ground_test = MyDataset(txt_path='test_cvbrct_street.txt', transform=transform_train)
        aerial_train = MyDataset(txt_path='tarin_cvbrct_air.txt', transform=transform_train)
        aerial_val = MyDataset(txt_path='cal_cvbrct_air.txt', transform=transform_train)
        aerial_test = MyDataset(txt_path='test_cvbrct_air.txt', transform=transform_train)
    else:
        ground_train = MyDataset(txt_path='tarin_airound_ground.txt', transform=transform_train)
        ground_val = MyDataset(txt_path='val_airound_ground.txt', transform=transform_train)
        ground_test = MyDataset(txt_path='test_airound_ground.txt', transform=transform_train)
        aerial_train = MyDataset(txt_path='tarin_airound_air.txt', transform=transform_train)
        aerial_val = MyDataset(txt_path='val_airound_air.txt', transform=transform_train)
        aerial_test = MyDataset(txt_path='test_airound_air.txt', transform=transform_train)

    ground_train_loader  = DataLoader(ground_train, batch_size=opt.batch_size)
    ground_val_loader  = DataLoader(ground_val, batch_size=opt.batch_size)
    ground_test_loader  = DataLoader(ground_test, batch_size=opt.batch_size)
    aerial_train_loader  = DataLoader(aerial_train, batch_size=opt.batch_size)
    aerial_val_loader  = DataLoader(aerial_val, batch_size=opt.batch_size)
    aerial_test_loader  = DataLoader(aerial_test, batch_size=opt.batch_size)

    # data_process
    g_train_feature, g_train_target = data_process(model, ground_train_loader)
    g_val_feature, g_val_target = data_process(model, ground_val_loader)
    g_test_feature, g_tset_target = data_process(model, ground_test_loader)
    a_train_feature, a_train_target = data_process(model, aerial_train_loader)
    a_val_feature, a_val_target = data_process(model, aerial_val_loader)
    a_test_feature, a_tset_target = data_process(model, aerial_test_loader)
    
    savemat('airound_data.mat', {'x1_train':g_train_feature, 
                                 'a_g_l_train': g_train_target,
                                 'x1_test':g_test_feature, 
                                 'a_g_l_test': g_tset_target,
                                 'x2_train':a_train_feature, 
                                 'a_a_l_train': a_train_target,
                                 'x2_test':a_test_feature, 
                                 'a_a_l_test': a_tset_target}) 


if __name__ == "__main__":
    main()
    