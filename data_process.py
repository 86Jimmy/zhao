import torch 
import argparse
from pathlib import Path
from torch.autograd import Variable
from model import MVGG
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from  utils import MyDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--view', type=int, default=2, help='number of sense view')
    parser.add_argument('--classes', type=int, default=9, help='number of class')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--train_txt', type=str, default= ROOT / 'train_list.txt' , 
                        help='path of train_data')
    parser.add_argument('--data', type=str, default= 'E:\\Document\\stage1\\data\\airound\\aerial' , 
                        help='path of train_data')
    parser.add_argument('--test_txt', type=str, default= ROOT / 'test_list.txt' , 
                        help='path of test_data')
    
    opt = parser.parse_args(args=[])

    return opt

def evalute(model, loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            target = Variable(target.long().cuda())
            model(data, target)

def main():
    opt = parse_option()
    if torch.cuda.is_available():
        print('gpu个数:', torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print('gpu名称:', torch.cuda.get_device_name(idx))    

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_data = MyDataset(txt_path=opt.train_txt, data_path=opt.data)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, transform=)    
    test_data = MyDataset(txt_path=opt.test_txt, data_path=opt.data)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size)  
    
    # model = MVGG()    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # 推断
    # for epoch in range(opt.epoches):
    #     evalute(model, test_loader)





    
if __name__=="__main__":
    main()




