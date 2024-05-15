# 判断两个文档之间的差异

txt_path = ['tarin_cvbrct_street.txt', 'val_cvbrct_street.txt','test_cvbrct_street.txt',
            'tarin_cvbrct_air.txt', 'val_cvbrct_air.txt', 'test_cvbrct_air.txt',
			'tarin_airound_ground.txt', 'val_airound_ground.txt', 'test_airound_ground.txt',
            'tarin_airound_air.txt', 'val_airound_air.txt','test_airound_air.txt']

def check(txt1, txt2):      
    fh1 = open(txt1, 'r') 
    fh2 = open(txt2, 'r')    
    fh = dict(zip(fh1, fh2))             
    for line1, line2 in fh.items():
        line1, line2 = line1.rstrip(), line2.rstrip()  # 将文章最后一个换行符去掉
        words1, words2 = line1.split(), line2.split()  # 
        words1[0] = words1[0][:-4].split("/",7)[7]
        words2[0] = words2[0][:-4].split("/",7)[7]
        if (words1[0] != words2[0]) or (words1[1] != words2[1]):
            print(txt1 , words1)    
            print(txt2 , words2)    
            return 0
    return 1    

def main():
    data_set_nums, data_sets = 2, 3
    for data_set_num in range(data_set_nums):
        for data_set in range(data_sets):
            ans  = check(txt_path[data_set_num*6+data_set], txt_path[data_set_num*6+3+data_set])
            print(txt_path[data_set_num*6+data_set], txt_path[data_set_num*6+3+data_set])
            if ans:
                print("smae")
            else:
                print("different")



if __name__ == "__main__":
    main()