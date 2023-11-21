import torch
import numpy as np
from robustbench import load_model
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from zoutendijk_attack import zoutendijk_attack_Linfty_mode1,zoutendijk_attack_Linfty_mode2,zoutendijk_attack_L2
import matplotlib.pyplot as plt
import time
import torchattacks

random_seed = 1
torch.manual_seed(random_seed)
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

test_dataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms.ToTensor())
criterion=nn.CrossEntropyLoss()

list_para=[
{'model':'Standard','attack':'Z_linf_mode1','epsilon':8.0/255,'stepsize':0.005,'delta':0.0075,'iter_max':20},
{'model':'Standard','attack':'Z_linf_mode2','epsilon':8.0/255,'delta':0.0075,'iter_max':20,'div':2.0},
{'model':'Standard','attack':'PGD_linf','epsilon':8.0/255,'stepsize':0.005,'delta':0.0075,'iter_max':20},
{'model':'Standard','attack':'MIFGSM_inf','epsilon':8.0/255,'stepsize':0.01,'iter_max':100},

{'model':'Wong','attack':'Z_linf_mode1','epsilon':8.0/255,'stepsize':0.005,'delta':0.0075,'iter_max':100},
{'model':'Wong','attack':'Z_linf_mode2','epsilon':8.0/255,'delta':0.0075,'iter_max':100,'div':2.0},
#{'model':'Wong','attack':'Z_l2','epsilon':0.5,'stepsize':0.01,'delta':0.015,'iter_max':20},
{'model':'Wong','attack':'PGD_linf','epsilon':8.0/255,'stepsize':0.005,'delta':0.0075,'iter_max':20},
{'model':'Wong','attack':'MIFGSM_linf','epsilon':8.0/255,'stepsize':0.01,'iter_max':100},

{'model':'Wang','attack':'Z_linf_mode1','epsilon':0.1,'stepsize':0.01,'delta':0.015,'iter_max':50},
{'model':'Wang','attack':'Z_linf_mode2','epsilon':0.1,    'delta':0.015,'iter_max':50,'div':1.5},
#{'model':'Wang','attack':'Z_l2','epsilon':0.5,'stepsize':0.01,'delta':0.015,'iter_max':20}
{'model':'Wang','attack':'PGD_linf','epsilon':0.1,'stepsize':0.01,'iter_max':50},
{'model':'Wang','attack':'MIFGSM_inf','epsilon':0.1,'stepsize':0.01,'iter_max':50}
]

for item in list_para:
   list1_success=[]
   list_diff=[]
   list_num=[]
   list_loss=[]
   start_time = time.time()
   print(item)
   if item['model'] == 'Standard':
       model = load_model(model_name='Standard', dataset='cifar10', norm='Linf')
   elif item['model'] == 'Wong':
       model = load_model(model_name='Wong2020Fast', norm='Linf')
   elif item['model'] == 'Wang':
       model = load_model(model_name='Wang2023Better_WRN-70-16', dataset='cifar10', norm='L2')
   model.to(device)
   model.eval()

   test_data = torch.tensor(test_dataset.data, device=device)
   test_labels = torch.tensor(test_dataset.targets, device=device)
   test_data_normalized = test_data / 255.0
   test_data_normalized = test_data_normalized
   test_data_normalized = test_data_normalized.permute(0, 3, 1, 2)
   test_accuracy = False
   if test_accuracy == True:
       predict_result = torch.tensor([], device=device)
       for i in range(500):
           outputs = model(test_data_normalized[20 * i:20 * i + 20])
           _, labels_predict = torch.max(outputs, 1)
           predict_result = torch.cat((predict_result, labels_predict), dim=0)
       correct = torch.eq(predict_result, test_labels)
       torch.save(correct, './result/cifar10/{}_Cifar_correct_predict.pt'.format(item['model']))
   else:
       correct = torch.load('./result/cifar10/{}_Cifar_correct_predict.pt'.format(item['model']))
   clean_accuracy = correct.sum() / 10000.0
   print('model clean accuracy:', clean_accuracy)

   for i in range(10000):
       print(i)
       if correct[i]:
           image=torch.unsqueeze(test_data_normalized[i],dim=0)

           #for showing images
           #outputs = model_Standard(image)
           #_, labels_predict = torch.max(outputs, 1)
           #imshow(torchvision.utils.make_grid(image.cpu().data, normalize=True), 'Predict:{}'.format(labels_predict.item()))

           label=torch.unsqueeze(test_labels[i],dim=0)
           if item['attack']=='Z_linf_mode1':
              success, predict_label, adv_image,iter,loss_eva = zoutendijk_attack_Linfty_mode1(model, image, label, epsilon=item['epsilon'],iter_max=item['iter_max'], delta=item['delta'],stepsize_default=item['stepsize'])
           elif item['attack']=='Z_linf_mode2':
               success, predict_label,adv_image,iter,loss_eva = zoutendijk_attack_Linfty_mode2(model, image, label, epsilon=item['epsilon'],iter_max=item['iter_max'],delta=item['delta'],div=item['div'])
           elif item['attack']=='Z_l2':
               success, predict_label,adv_image,iter,loss_eva = zoutendijk_attack_L2(model, image, label, epsilon=item['epsilon'],iter_max=item['iter_max'],delta=item['delta'],stepsize_default=item['stepsize'])
           elif item['attack']=='PGD_linf':
               atk = torchattacks.PGD(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'])  # torchattack
               adv_image, perturbation, num, success, loss_eva, predict_label = atk(image, label)
           elif item['attack']=='MIFGSM_linf':
               atk = torchattacks.MIFGSM(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'])  # torchattack
               adv_image, perturbation, num, success, loss_eva, predict_label = atk(image, label)

           if success:
              list1_success.append(i)
              if 'linf' in item['attack']:
                  list_diff.append(torch.max(torch.abs(adv_image - image)))
              elif 'l2' in item['attack']:
                  list_diff.append(torch.norm(adv_image - image, p=2))
              list_num.append(iter)
              list_loss.append(loss_eva)
              #imshow(torchvision.utils.make_grid(adv_image.cpu().data, normalize=True), 'Predict:{}'.format(predict_label.item()))
           print(success)
   end_time = time.time()
   time_used=end_time-start_time
   attack_success_rate=len(list1_success)/correct.sum()
   print('attack success rate:',attack_success_rate)
   dict_save={'clean_accu':clean_accuracy,'para':item,'time_used':time_used,'list1_success':list1_success,'attack_success_rate':attack_success_rate,'list_diff':list_diff,'list_num':list_num,'list_loss':list_loss}
   if 'Z_' in item['attack']:
       torch.save(dict_save, './result/cifar10/{}_attack_{}_epsilon{}_delta{}.pt'.format(item['model'], item['attack'],item['epsilon'], item['delta']))
   elif 'PGD' in item['attack'] or 'MIFGSM' in item['attack']:
       torch.save(dict_save,'./result/cifar10/{}_attack_{}_epsilon{}.pt'.format(item['model'], item['attack'], item['epsilon']))
   #dict_read=torch.load('./result/cifar10/StandardCifar_attack_Z_linf_mode1_epsilon0.03137254901960784_delta0.0075.pt')
