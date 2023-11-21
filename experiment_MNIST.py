import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_MNIST import model_MNIST
from zoutendijk_attack import zoutendijk_attack_Linfty_mode1,zoutendijk_attack_Linfty_mode2,zoutendijk_attack_L2
import time
import matplotlib.pyplot as plt
import torchattacks
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

test_dataset=datasets.MNIST(root='../data',
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

model=model_MNIST()
model.to(device)
criterion=nn.CrossEntropyLoss()

list_para=[
{'model':'Standard','attack':'Z_linf_mode1','epsilon':0.2,'stepsize':0.01,'delta':0.015,'iter_max':50},
{'model':'Standard','attack':'Z_linf_mode1','epsilon':0.3,'stepsize':0.01,'delta':0.015,'iter_max':100},
{'model':'Standard','attack':'Z_linf_mode1','epsilon':0.4,'stepsize':0.01,'delta':0.015,'iter_max':50},
{'model':'Standard','attack':'Z_linf_mode2','epsilon':0.2,'delta':0.015,'iter_max':100,'div':2.0},
{'model':'Standard','attack':'Z_linf_mode2','epsilon':0.3,'delta':0.015,'iter_max':100,'div':2.0},
#{'model':'Standard','attack':'Z_l2','epsilon':2.0,'stepsize':0.02,'delta':0.25,'iter_max':100},
{'model':'ddn','attack':'Z_linf_mode1','epsilon':0.3,'stepsize':0.01,'delta':0.015,'iter_max':200},
{'model':'ddn','attack':'Z_linf_mode1','epsilon':0.4,'stepsize':0.01,'delta':0.015,'iter_max':200},
{'model':'ddn','attack':'Z_linf_mode2','epsilon':0.3,'delta':0.015,'iter_max':500,'div':1.2},
#{'attack':'Z_l2','epsilon':0.5,'stepsize':0.01,'delta':0.015,'iter_max':40},
{'model':'trades','attack':'Z_linf_mode1','epsilon':0.4,'stepsize':0.01,'delta':0.015,'iter_max':200},
{'model':'trades','attack':'Z_linf_mode2','epsilon':0.4,'delta':0.015,'iter_max':300,'div':1.2},
#{'attack':'Z_l2','epsilon':0.5,'stepsize':0.01,'delta':0.015,'iter_max':40},

{'model':'Standard','attack':'PGD_linf','epsilon':0.2,'stepsize':0.01,'iter_max':50},
{'model':'Standard','attack':'PGD_linf','epsilon':0.3,'stepsize':0.01,'iter_max':50},
{'model':'Standard','attack':'PGD_linf','epsilon':0.4,'stepsize':0.01,'iter_max':50},
{'model':'ddn','attack':'PGD_linf','epsilon':0.3,'stepsize':0.01,'iter_max':200},
{'model':'ddn','attack':'PGD_linf','epsilon':0.4,'stepsize':0.01,'iter_max':200},
{'model':'trades','attack':'PGD_linf','epsilon':0.4,'stepsize':0.01,'iter_max':200},

{'model':'Standard','attack':'MIFGSM_linf','epsilon':0.2,'stepsize':0.01,'iter_max':100},
{'model':'Standard','attack':'MIFGSM_linf','epsilon':0.3,'stepsize':0.01,'iter_max':100},
{'model':'Standard','attack':'MIFGSM_linf','epsilon':0.4,'stepsize':0.01,'iter_max':100},
{'model':'ddn','attack':'MIFGSM_linf','epsilon':0.3,'stepsize':0.01,'iter_max':200},
{'model':'ddn','attack':'MIFGSM_linf','epsilon':0.4,'stepsize':0.01,'iter_max':200},
{'model':'trades','attack':'MIFGSM_linf','epsilon':0.4,'stepsize':0.01,'iter_max':200}
]

for item in list_para:
    list_success=[]
    list_diff=[]
    list_num=[]
    list_loss=[]
    start_time = time.time()
    print(item)
    for i in range(10000):
        print('***************{}th data***********'.format(i))
        if item['model']=='Standard':
            model.load_state_dict(torch.load('mnist_regular.pth'), False)
        elif item['model']=='ddn':
            model.load_state_dict(torch.load('mnist_robust_ddn.pth'), False)  # ALM paper github : l2 adversarially trained
        elif  item['model']=='trades':
            model.load_state_dict(torch.load('mnist_robust_trades.pt'), False)  # ALM paper github: l_\infty adversarially trained
        model.eval()  # turn off the dropout
        test_data = torch.unsqueeze(test_dataset.data, dim=1)
        test_labels = test_dataset.test_labels.to(device)
        test_data_normalized = test_data / 255.0
        test_data_normalized = test_data_normalized.to(device)
        outputs = model(test_data_normalized)
        _, labels_predict = torch.max(outputs, 1)
        correct = torch.eq(labels_predict, test_labels)
        correct_sum = correct.sum()
        print(correct.sum())
        print('clean accuracy is:', correct.sum() / 10000.0)

        if correct[i]:
            image=torch.unsqueeze(test_data_normalized[i],dim=0).clone().detach()
            label=torch.unsqueeze(test_labels[i],dim=0)
            #for showing image
            #outputs = model(image)
            #_, labels_predict = torch.max(outputs, 1)
            #imshow(torchvision.utils.make_grid(image.cpu().data, normalize=True), 'Predict:{}'.format(labels_predict.item()))

            if item['attack']=='Z_linf_mode1':
               success,predict_label,adv_image,iter,loss_eva=zoutendijk_attack_Linfty_mode1(model,image,label,epsilon=item['epsilon'],iter_max=item['iter_max'],delta=item['delta'],stepsize_default=item['stepsize'])
            elif item['attack'] == 'Z_linf_mode2':
               success, predict_label,adv_image,iter,loss_eva = zoutendijk_attack_Linfty_mode2(model, image, label,epsilon=item['epsilon'],iter_max=item['iter_max'],delta=item['delta'],div=item['div'])
            elif item['attack'] == 'Z_l2':
               success, predict_label,adv_image,iter,loss_eva = zoutendijk_attack_L2(model, image, label,epsilon=item['epsilon'],iter_max=item['iter_max'],delta=item['delta'],stepsize_default=item['stepsize'])
            elif item['attack']=='PGD_linf':
                atk = torchattacks.PGD(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'])  # torchattack
                adv_image, perturbation, num, success, loss_eva, predict_label = atk(image, label)
            elif item['attack']=='MIFGSM_linf':
                atk = torchattacks.MIFGSM(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'])  # torchattack
                adv_image, perturbation, num, success, loss_eva, predict_label = atk(image, label)
            if 'l2' in item['attack']:
                print('distortion norm2((images-images_ori),p=2):', torch.norm(adv_image - image,p=2))
            elif 'linf' in item['attack']:
                print('distortion max(|images-images_ori|):', torch.max(torch.abs(adv_image - image)))
            if success:
               list_success.append(i)
               list_diff.append(torch.max(torch.abs(adv_image - image)))
               list_num.append(iter)
               list_loss.append(loss_eva)
               #imshow(torchvision.utils.make_grid(adv_image.cpu().data, normalize=True), 'Predict:{}'.format(predict_label.item()))
            print(success)
    end_time = time.time()
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    attack_success_rate=len(list_success)/correct_sum
    print('attack success rate:',attack_success_rate)
    dict_save={'para':item,'time_used':time_used,'list1_success':list_success,'attack_success_rate':attack_success_rate,'list_diff':list_diff,'list_num':list_num,'list_loss':list_loss}
    if 'Z_' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}_delta{}.pt'.format(item['model'],item['attack'],item['epsilon'],item['delta']))
    elif 'PGD' in item['attack'] or 'MIFGSM' in item['attack']:
       torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}.pt'.format(item['model'], item['attack'], item['epsilon']))

