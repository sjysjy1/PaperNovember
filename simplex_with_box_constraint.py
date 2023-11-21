import numpy as np
class simplex_with_box_constraint(object):
    def __init__(self,A,b,c,lower_bound,upper_bound,basis_var,nonbasis_var,lower_bound_var,upper_bound_var,phase):
        self.A = A
        self.b = b
        self.c = c
        self.lower_bound=lower_bound
        self.upper_bound = upper_bound
        self.basis_var = basis_var
        self.nonbasis_var = nonbasis_var
        self.lower_bound_var = lower_bound_var
        self.upper_bound_var = upper_bound_var
        self.phase=phase


    def find_solution(self):
        B = self.A[:, self.basis_var]
        N = self.A[:, self.nonbasis_var]
        x = np.mat([[0] for i in range(self.A.shape[1])], dtype=float)
        x[self.lower_bound_var] =self.lower_bound[self.lower_bound_var]
        x[self.upper_bound_var] =self.upper_bound[self.upper_bound_var]
        x[self.basis_var] = np.linalg.inv(B)*(self.b - np.dot(N, x[self.nonbasis_var]))
        b_hat = np.linalg.inv(B)*(self.b - np.dot(N, x[self.nonbasis_var]))
        self.A=np.linalg.inv(B)*self.A
        objective_value = np.dot(self.c, x)
        minus_reduced_cost = -self.c
        for i in self.basis_var:
              if minus_reduced_cost[0,i]!=0:
                  minus_reduced_cost[0]=minus_reduced_cost[0]-minus_reduced_cost[0,i]*self.A[self.basis_var.index(i)]

        count=0
        while np.any(minus_reduced_cost[0,self.lower_bound_var]>0) or np.any(minus_reduced_cost[0,self.upper_bound_var]<0):
               print('current objective value is(should be decreasing):',self.c*x)
               count=count+1
               if (np.any(minus_reduced_cost[0, self.lower_bound_var] > 0)):
                  print(np.argmax(minus_reduced_cost[0, self.lower_bound_var]))
                  pivot_column_index = self.lower_bound_var[np.argmax(minus_reduced_cost[0,self.lower_bound_var])]
                  y=self.A[:,pivot_column_index]
                  if np.all(y<=0):
                      gamma_1=np.inf
                  else:
                      #print(lower_bound[0])
                      list_gamma_1=[[(b_hat[i,0]-self.lower_bound[self.basis_var[i],0])/y[i,0],i] for i in range(len(y)) if y[i]>0]
                      gamma_1=np.min(np.array(list_gamma_1)[:,0])
                      gamma_1_which=np.argwhere(np.array(list_gamma_1)[:,0]==gamma_1)
                      leave_basis_index_g1=list_gamma_1[gamma_1_which[0][0]][1]

                  if np.all(y>=0):
                      gamma_2=np.inf
                  else:
                      list_gamma_2=[[(self.upper_bound[self.basis_var[i],0]-b_hat[i,0])/(-y[i,0]),i] for i in range(len(y)) if y[i]<0]
                      gamma_2=np.min(np.array(list_gamma_2)[:,0])
                      gamma_2_which = np.argwhere(np.array(list_gamma_2)[:,0]==gamma_2)
                      #if len(gamma_2_which)!= 1:
                      #    print('Warning: gamma_2 is determined by more than one iterm')
                      if np.isnan(gamma_2):
                          print('gamma_2 is nan!!!')
                      leave_basis_index_g2 = list_gamma_2[gamma_2_which[0][0]][1]
                  Delta=np.min([gamma_1,gamma_2,self.upper_bound[pivot_column_index,0]-self.lower_bound[pivot_column_index,0]])
                  if Delta==np.inf:
                      print('This problem is unbounded!??')
                  which=np.argwhere([gamma_1,gamma_2,self.upper_bound[pivot_column_index,0]-self.lower_bound[pivot_column_index,0]]==Delta)
                  if len(which) != 1:
                      print('Warning: Delta is determined by more than one iterm')
                  x[self.basis_var] = b_hat - y * Delta
                  if b_hat[0]==np.inf:
                      print('b_hat[0]==np.inf is true!!')
                  x[pivot_column_index] = x[pivot_column_index] + Delta  # only one nonbasis var is changed
                  if 0 in which and 1 in which:
                      leave_basis_index = min(leave_basis_index_g1,leave_basis_index_g1) # to be modified
                  elif 0 in which:
                      leave_basis_index = leave_basis_index_g1
                  elif 1 in which:
                      leave_basis_index = leave_basis_index_g2
                  if 0 in which or 1 in which:
                     leave_basis_var = self.basis_var[leave_basis_index]
                  pivot_var = pivot_column_index
                  self.lower_bound_var.remove(pivot_var)
                  if 0 in which : #one variable leaves the basis and drops to lower bound
                      if leave_basis_var == self.basis_var[list_gamma_1[gamma_1_which[0][0]][1]]:
                         self.lower_bound_var.append(leave_basis_var)
                  if 1 in which: #one variable leaves the basis and reaches upper bound
                      #leave_basis_index = leave_basis_index_g2
                      if leave_basis_var == self.basis_var[list_gamma_2[gamma_2_which[0][0]][1]]:
                          self.upper_bound_var.append(leave_basis_var)
                      self.upper_bound_var.sort()
                      #print('null')
                  if 0 in which or 1 in which:
                      self.basis_var[leave_basis_index]=pivot_var
                      self.nonbasis_var.remove(pivot_var)
                      self.nonbasis_var.append(leave_basis_var)
                      self.nonbasis_var.sort()
                  if 2 in which and len(which) == 1:
                      self.upper_bound_var.append(pivot_var)
                      self.upper_bound_var.sort()
                  objective_value= objective_value-minus_reduced_cost[0,pivot_column_index]*Delta
                  b_hat=x[self.basis_var]

               elif np.any(minus_reduced_cost[0,self.upper_bound_var]<0):
                  pivot_column_index = self.upper_bound_var[np.argmin(minus_reduced_cost[0, self.upper_bound_var])]
                  y=self.A[:,pivot_column_index]
                  if np.all(y>=0):
                      gamma_1=np.inf
                  else:
                      list_gamma_1=[[(b_hat[i,0]-self.lower_bound[self.basis_var[i],0])/(-y[i,0]),i] for i in range(len(y)) if y[i]<0]
                      gamma_1=np.min(np.array(list_gamma_1)[:,0])
                      gamma_1_which=np.argwhere(np.array(list_gamma_1)[:,0]==gamma_1)
                      if len(gamma_1_which)!= 1:
                          print('Warning: gamma_1 is determined by more than one item')
                      leave_basis_index_g1=list_gamma_1[gamma_1_which[0][0]][1]
                  if np.all(y<=0):
                      gamma_2=np.inf
                  else:
                      list_gamma_2=[[(self.upper_bound[self.basis_var[i],0]-b_hat[i,0])/y[i,0],i] for i in range(len(y)) if y[i]>0]
                      gamma_2=np.min(np.array(list_gamma_2)[:,0])
                      gamma_2_which = np.argwhere(np.array(list_gamma_2)[:,0]==gamma_2)
                      if len(gamma_2_which) != 1:
                          print('Warning: gamma_2 is determined by more than one item')
                      leave_basis_index_g2 = list_gamma_2[gamma_2_which[0][0]][1]
                  Delta=np.min([gamma_1,gamma_2,self.upper_bound[pivot_column_index,0]-self.lower_bound[pivot_column_index,0]])
                  which=np.argwhere([gamma_1,gamma_2,self.upper_bound[pivot_column_index,0]-self.lower_bound[pivot_column_index,0]]==Delta)
                  if len(which)!=1:
                      print('Warning: delta is determined by more than one iterm')
                  x[self.basis_var] = b_hat + y * Delta
                  x[pivot_column_index] = x[pivot_column_index] - Delta  # only one nonbasis var is changed
                  if 0 in which and 1 in which:
                      leave_basis_index = min(leave_basis_index_g1,leave_basis_index_g1) # to be modified
                  elif 0 in which:
                      leave_basis_index = leave_basis_index_g1
                  elif 1 in which:
                      leave_basis_index = leave_basis_index_g2
                  if 0 in which or 1 in which:
                     leave_basis_var = self.basis_var[leave_basis_index]
                  pivot_var = pivot_column_index
                  self.upper_bound_var.remove(pivot_var)
                  if 0 in which: #one variable leavs the basis and drops to lower bound
                      if leave_basis_var ==self.basis_var[list_gamma_1[gamma_1_which[0][0]][1]]:
                         self.lower_bound_var.append(self.basis_var[list_gamma_1[gamma_1_which[0][0]][1]])

                  if 1 in which: #one variable leaves the basis and reaches upper bound
                      if leave_basis_var == self.basis_var[list_gamma_2[gamma_2_which[0][0]][1]]:
                          self.upper_bound_var.append(leave_basis_var)
                          self.upper_bound_var.sort()

                  if 0 in which or 1 in which:
                      self.basis_var[leave_basis_index]=pivot_var
                      self.nonbasis_var.remove(pivot_var)
                      self.nonbasis_var.append(leave_basis_var)
                      self.nonbasis_var.sort()
                  if 2 in which and len(which) == 1:
                      self.lower_bound_var.append(pivot_var)
                      self.lower_bound_var.sort()

                  objective_value= objective_value+minus_reduced_cost[0,pivot_column_index]*Delta
                  b_hat=x[self.basis_var]
                  #update the simplex table
               if 0 in which or 1 in which:
                  self.A[leave_basis_index,:]=self.A[leave_basis_index,:]/self.A[leave_basis_index,pivot_column_index]
                  #print(self.A)
                  multiplier=minus_reduced_cost[0,pivot_column_index]
                  minus_reduced_cost[0] = minus_reduced_cost[0] - multiplier * self.A[leave_basis_index]

                  #print(minus_reduced_cost)
                  for i in range(self.A.shape[0]):
                      if i!= leave_basis_index:
                          multiplier = self.A[i, pivot_column_index]
                          self.A[i] = self.A[i] - self.A[leave_basis_index] * multiplier
        return objective_value,x,self.A,b_hat,self.basis_var,self.nonbasis_var,self.lower_bound_var,self.upper_bound_var


