import scipy.io as scio

data=scio.loadmat('cvbrct_data.mat')

print(type(data))
# print(data['x1_l_val'])

if data['x1_l_train'].all() == data['x2_l_train'].all():
    print("train is same")

if data['x1_l_val'].all() == data['x2_l_val'].all():
    print("val is same")

if data['x1_l_test'].all() == data['x2_l_test'].all():
    print("test is same")