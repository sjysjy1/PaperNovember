import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from simplex_with_box_constraint import simplex_with_box_constraint
def loss(model,images,labels):
    #optimizer.zero_grad()
    outputs = model(images)
    CELoss = nn.CrossEntropyLoss()
    loss = -CELoss(outputs, labels)
    return loss

def zoutendijk_attack_Linfty_mode1(model:nn.Module,
                             images:Tensor,
                             labels:Tensor,
                             epsilon:float=0.2,
                             stepsize_default:float=0.01,
                             delta:float=0.015,
                             iter_max:int=20,
                             targeted: bool = False)-> Tensor:
    # Zoutendijk_attack: L_infty
    iter=0
    device=images.device
    one_tensor=torch.ones(images.shape, dtype=torch.float64, device=device)
    zero_tensor = torch.zeros(images.shape, dtype=torch.float64, device=device)
    upper_bound = torch.min(images + epsilon,one_tensor)
    lower_bound = torch.max(images - epsilon,zero_tensor)
    attack_num_show=[]
    attack_loss_show=[]
    success=False
    while iter<iter_max:
        iter+=1
        attack_num_show.append(iter)
        print('iter=',iter)
        images.requires_grad_()
        model.zero_grad()
        loss_eva = loss(model, images, labels)
        attack_loss_show.append(loss_eva.detach().cpu().numpy())
        print('current loss is:loss_eva=', loss_eva)
        loss_eva.backward()
        direction=torch.zeros(images.shape,dtype=torch.float,device=device)
        #general Zoutendijk's method
        flag1=(images.grad>=0) & (images > lower_bound+delta)
        flag2=(images.grad < 0) & (images<upper_bound-delta)
        for c in range(images.shape[1]):
           for i in range(images.shape[2]):
               for j in range(images.shape[3]):
                 if flag1[0][c][i][j]:
                     direction[0][c][i][j]=-1
                 elif flag2[0][c][i][j]:
                     direction[0][c][i][j]=1
        stepsize_max=epsilon
        for c in range(images.shape[1]):
            for i in range(images.shape[2]):
                for j in range(images.shape[3]):
                   if (direction[0][c][i][j]==-1) & (stepsize_max>images[0][c][i][j]-lower_bound[0][c][i][j]):
                       stepsize_max=images[0][c][i][j]-lower_bound[0][c][i][j]#(upper_bound[0][c][i][j]-images[0][c][i][j])
                   elif (direction[0][c][i][j]==1) & (stepsize_max>upper_bound[0][c][i][j]-images[0][c][i][j]):
                       stepsize_max =upper_bound[0][c][i][j]-images[0][c][i][j]#images[0][c][i][j]-lower_bound[0][c][i][j]
        stepsize = stepsize_default
        print('stepsize_max is:', stepsize_max)
        if stepsize>stepsize_max:
            stepsize=stepsize_max
        print('stepsize is:', stepsize)
        images = (images + stepsize * direction).detach()
        outputs = model(images)
        _, predictions_attacked = torch.max(outputs, 1)
        if predictions_attacked != labels:
          success=True
          break
    return success, predictions_attacked,images,iter,loss_eva



def zoutendijk_attack_Linfty_mode2(model:nn.Module,
                             images:Tensor,
                             labels:Tensor,
                             epsilon:float=0.2,
                             #stepsize_default:float=0.01,
                             delta:float=0.015,
                             iter_max:int=100,
                             div:float=2.0,
                             targeted: bool = False)-> Tensor:
    # Zoutendijk_attack: L_infty
    iter=0
    device = images.device
    one_tensor=torch.ones(images.shape, dtype=torch.float64, device=device)
    zero_tensor = torch.zeros(images.shape, dtype=torch.float64, device=device)
    upper_bound = torch.min(images + epsilon,one_tensor)
    lower_bound = torch.max(images - epsilon,zero_tensor)
    attack_num_show=[]
    attack_loss_show=[]
    success=False
    while iter<iter_max:
        iter+=1
        attack_num_show.append(iter)
        print('iter=',iter)
        images.requires_grad_()
        model.zero_grad()
        loss_eva = loss(model, images, labels)
        attack_loss_show.append(loss_eva.detach().cpu().numpy())
        print('current loss is:loss_eva=', loss_eva.item())
        loss_eva.backward()
        direction=torch.zeros(images.shape,dtype=torch.float,device=device)
        #general Zoutendijk's method
        flag1=(images.grad > 0) & (images > lower_bound+delta)
        flag2=(images.grad < 0) & (images<upper_bound-delta)
        flag= flag1 | flag2
        hardmard_product=torch.mul(images[0],flag[0])
        square_sum=(hardmard_product.view(-1,hardmard_product.shape[0]*hardmard_product.shape[1]*hardmard_product.shape[2]) ** 2).sum(-1).item()
        for c in range(images.shape[1]):
          for i in range(images.shape[2]):
            for j in range(images.shape[3]):
               if flag[0][c][i][j]:
                    direction[0][c][i][j]=-images.grad[0][c][i][j]/square_sum**0.5
        stepsize_max=np.infty
        for c in range(images.shape[1]):
          for i in range(images.shape[2]):
            for j in range(images.shape[3]):
                temp1=(images[0][c][i][j] - lower_bound[0][c][i][j]) / torch.abs(-direction[0][c][i][j])
                temp2=(upper_bound[0][c][i][j] - images[0][c][i][j]) / torch.abs(direction[0][c][i][j])
                if (flag1[0][c][i][j]) & (stepsize_max>temp1):
                   stepsize_max=temp1
                elif (flag2[0][c][i][j]) & (stepsize_max>temp2):
                   stepsize_max =temp2
        stepsize=stepsize_max/div
        images = (images + stepsize * direction).detach()
        outputs = model(images)
        _, predictions_attacked = torch.max(outputs, 1)
        if predictions_attacked != labels:
          success=True
          break
    return success, predictions_attacked,images,iter,loss_eva



def zoutendijk_attack_L2(model:nn.Module,
                             images:Tensor,
                             labels:Tensor,
                             epsilon:float=0.5,
                             stepsize_default:float=0.01,
                             iter_max:int=100,
                             delta:float=0.015,
                             targeted: bool = False)-> Tensor:
    # Zoutendijk_attack: L_2
    iter=0
    device = images.device
    images_original=images.clone().detach()
    addr1=images.data_ptr()
    addr2 = images.data_ptr()
    shape=images.shape
    one_tensor=torch.ones(images.shape, dtype=torch.float64, device=device)
    zero_tensor = torch.zeros(images.shape, dtype=torch.float64, device=device)
    upper_bound = one_tensor
    lower_bound = zero_tensor
    attack_num_show=[]
    success=False
    while iter<iter_max:
        iter+=1
        attack_num_show.append(iter)
        print('iter=',iter)
        images.requires_grad_()
        model.zero_grad()
        loss_eva = loss(model, images, labels)
        loss_eva.backward()
        direction=torch.zeros(images.shape,dtype=torch.float,device=device)

        print('norm(image-image_ori) is: ',torch.norm(images-images_original,p=2))
        total_len=images.shape[1]*images.shape[2]*images.shape[3]
        if torch.norm(images-images_original,p=2)**2>=0.9*epsilon**2: #the norm constraint is activated
            #Phase I
            S1 = [i  for i in range(total_len)  if images.view(-1, total_len)[0][i] >= 1.0-delta ]
            S2 = [i  for i in range(total_len)  if images.view(-1, total_len)[0][i] < delta]
            lower_bound_simplex = np.matrix([[0] if i in S2 else [-1] for i in range(total_len) ], dtype=float)
            upper_bound_simplex = np.matrix([[0] if i in S1 else [1] for i in range(total_len) ], dtype=float)
            print('lower_bound_simplex.sum() is:',lower_bound_simplex.sum())
            print('upper_bound_simplex.sum() is:', upper_bound_simplex.sum())

            lower_bound_simplex_aug1=np.concatenate((lower_bound_simplex,np.matrix([[0],[0],[0],[0],[0],[0]],dtype=float)),axis=0)
            upper_bound_simplex_aug1= np.concatenate((upper_bound_simplex,np.matrix([[np.inf],[np.inf],[np.inf],[np.inf],[np.inf],[np.inf]],dtype=float)),axis=0)
            if np.asmatrix(images.grad.view(-1,total_len).to('cpu'))*lower_bound_simplex<=0:
                coeff_1 = 1
            else:
                coeff_1 = -1
            if np.asmatrix((2*(images.detach()-images_original).view(-1,total_len).to('cpu')))*lower_bound_simplex<=0:
                coeff_2 = 1
            else:
                coeff_2 = -1
            A1=np.asmatrix(torch.cat((images.grad.view(-1,total_len).to('cpu'),torch.tensor([[-1,1,1,0,coeff_1,0]])),1).detach().numpy())
            A2=np.asmatrix(torch.cat((2*(images-images_original).view(-1,total_len).to('cpu'),torch.tensor([[-1,1,0,1,0,coeff_2]])),1).detach().numpy())
            A=np.concatenate((A1,A2),axis=0)
            b = np.matrix([[0],[0]], dtype=float)
            c=np.zeros((1,total_len+4),dtype=float)
            tmp=np.matrix([[1,1]])
            c=np.concatenate((c,np.matrix([[1, 1]])),axis=1)
            basis_var = [A.shape[1]-2, A.shape[1]-1]  #x0,x1,x2....i.e. including index 0
            nonbasis_var = [i for i in range(A.shape[1]-2)]
            lower_bound_var=[i for i in range(A.shape[1]-2) ]
            upper_bound_var=[ ]
            print('*****************PhaseI*********************')
            direction_finding_phaseI = simplex_with_box_constraint(A, b, c, lower_bound_simplex_aug1, upper_bound_simplex_aug1,
                                                            basis_var, nonbasis_var,
                                                            lower_bound_var, upper_bound_var,phase=1)
            opt_value_phaseI,opt_sol_phaseI,A,b,basis_var,nonbasis_var,lower_bound_var,upper_bound_var = direction_finding_phaseI.find_solution()
            print('when PhaseI finishes, basis_var is (is artificial variable still in basis):',basis_var )
            #print('when PhaseI finishes, opt_sol_phaseI is :', opt_sol_phaseI)
            print('optimal value of PhaseI is (should be >=0):',opt_value_phaseI)

            #phase II
            c=np.concatenate((np.zeros((1,total_len),dtype=float),np.matrix([[1,-1,0,0]])),axis=1) # including the c of slack variables
            lower_bound_simplex_aug2=np.concatenate((lower_bound_simplex,np.matrix([[0],[0],[0],[0]],dtype=float)),axis=0)
            upper_bound_simplex_aug2= np.concatenate((upper_bound_simplex,np.matrix([[np.inf],[np.inf],[np.inf],[np.inf]],dtype=float)),axis=0)

            if total_len+4 in basis_var:
                basis_var.remove(total_len+4)
            if total_len+5 in basis_var:
                basis_var.remove(total_len + 5)
            if total_len+4 in nonbasis_var:
                nonbasis_var.remove(total_len+4)
            if total_len+5 in nonbasis_var:
               nonbasis_var.remove(total_len + 5)
            lower_bound_var.remove(total_len+4)
            lower_bound_var.remove(total_len + 5)
            print('*****************PhaseII*********************')
            direction_finding_phaseII = simplex_with_box_constraint(A[:,0:total_len+4], b, c, lower_bound_simplex_aug2, upper_bound_simplex_aug2, basis_var, nonbasis_var,
                                                 lower_bound_var, upper_bound_var,phase=2)
            opt_value,direction,_,_,_,_,_,_=direction_finding_phaseII.find_solution()
            direction=torch.FloatTensor(direction[0:total_len,0]).view(1,shape[1],shape[2],shape[3]).to(device)
            print('optimal value of PhaseII is (should be <=0):',opt_value)

        else:
            flag1 = (images.grad > 0) & (images > lower_bound + delta)
            flag2 = (images.grad < 0) & (images < upper_bound - delta)
            for c in range(images.shape[1]):
                for i in range(images.shape[2]):
                    for j in range(images.shape[3]):
                        if flag1[0][c][i][j]:
                            # if flag1[0][0][i][j] == True:
                            direction[0][c][i][j] = -1
                        elif flag2[0][c][i][j]:
                            # elif flag2[0][0][i][j] == True:
                            direction[0][c][i][j] = 1
        discriminant=4*torch.matmul((images-images_original).view(-1,total_len),direction.view(-1,total_len).t())**2-4*(torch.norm(direction,p=2)*((torch.norm(images-images_original,p=2))**2-epsilon*epsilon))
        stepsize_max=(-2*torch.matmul((images-images_original).view(-1,total_len),direction.view(-1,total_len).t())+torch.sqrt(discriminant))/(2*torch.norm(direction,p=2)**2)
        print('stepsize_max for L2 norm constraint is:',stepsize_max)
        if torch.isnan(stepsize_max):
            print('stepsize_max is nan!!!!!!!')
        for c in  range(images.shape[1]):
            for i in range(images.shape[2]):
                for j in range(images.shape[3]):
                   if (direction[0][c][i][j]==1) & (stepsize_max>(upper_bound[0][c][i][j]-images[0][c][i][j])):
                       stepsize_max=(upper_bound[0][c][i][j]-images[0][c][i][j])
                   elif (direction[0][c][i][j]==-1) & (stepsize_max>(images[0][c][i][j]-lower_bound[0][c][i][j])):
                       stepsize_max =(images[0][c][i][j]-lower_bound[0][c][i][j])
        print('stepsize_max for all constraints is:', stepsize_max)
        stepsize=stepsize_default
        if stepsize>stepsize_max:
            stepsize=stepsize_max
        print('stepsize is:', stepsize)
        images = (images + stepsize * direction).clone().detach()
        outputs = model(images)
        _, predictions_attacked = torch.max(outputs, 1)
        if predictions_attacked != labels:
          success=True
          break
    return success, predictions_attacked,images,iter,loss_eva




