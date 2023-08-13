# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:07:51 2020

@author: Yuanhao
"""
import os
import pickle
import numpy as np
from math import sqrt

# Store information for each method
class summaryTable(object):
    """Computes and stores the average and current value"""

    def __init__(self,
                 Torder, methodName,
                 tableFiles, tab_dir):
        self.valMatches = [Torder+"_"+methodName+"_", "full_val_acc"]
        self.testMatches = [Torder+"_"+methodName+"_", "full_test_acc"]
        self.increaseMatches = [Torder+"_"+methodName+"_", "increase_size"]
        self.tables = {}    
        self.find(tableFiles)
        self.assign(tab_dir)
        self.calForget()
        self.calAverage()
        self.getOptimal()

    def find(self,tableFiles):
        f1 = 0
        f2 = 0
        f3 = 0
        for filename in tableFiles:
             if all([(filename.find(x) != (-1)) for x in self.valMatches]):
                 self.Cvalname = filename
                 f1 += 1
             if all([(filename.find(x) != (-1)) for x in self.testMatches]):
                 self.Ctestname = filename
                 f2 += 1
             if all([(filename.find(x) != (-1)) for x in self.increaseMatches]):
                 self.Cincreasename = filename
                 f3 += 1
        if  f1==0:
            raise Exception("ValMatches not found!")
        elif f1>1:
            raise Exception("ValMatches found multiple!")    
            
        if  f2==0:
            raise Exception("ValMatches not found!")
        elif f2>1:
            raise Exception("ValMatches found multiple!")    
            
        if  f3==0:
            raise Exception("ValMatches not found!")
        elif f3>1:
            raise Exception("ValMatches found multiple!")  
            
    def assign(self,tab_dir):  
        with open( os.path.join(tab_dir, self.Cvalname) , 'rb') as table_file:
            self.tables['val'] = pickle.load(table_file)  
        with open( os.path.join(tab_dir, self.Ctestname) , 'rb') as table_file:
            self.tables['test'] = pickle.load(table_file)  
        with open( os.path.join(tab_dir, self.Cincreasename) , 'rb') as table_file:
            self.tables['increase'] = pickle.load(table_file)  
            
        self.l1_hyp_list = [x for x in self.tables['val'].keys()]
        for _,tmp in self.tables['val'].items():
            self.reg_coef_list = [x for x in tmp.keys()]
            for _,tmp2 in tmp.items():
                self.repeat = tmp2.shape[0]
                self.n_task = tmp2.shape[1]
                break
            break 

            
    def calForget(self):
        n_task = self.n_task
        repeat = self.repeat
        self.forget = {}
        for l1_hyp in self.l1_hyp_list:
            self.forget[l1_hyp] = {}
            for reg_coef in self.reg_coef_list:
                self.forget[l1_hyp][reg_coef] = np.zeros((repeat,n_task))
                for r in range(repeat):
                    self.forget[l1_hyp][reg_coef][r,:] = np.diag(self.tables['test'][l1_hyp][reg_coef][r,:,:].reshape(n_task,n_task)) - \
                        self.tables['test'][l1_hyp][reg_coef][r,n_task-1,:n_task]
    
    def calAverage(self):
        n_task = self.n_task
        repeat = self.repeat
        
        # mean
        self.Asize_task = {}
        self.Aforget_task = {}
        self.Aval_task = {}
        self.Atest_task = {}
        
        self.Asize = {}
        self.Aforget = {}
        self.Aval = {}
        self.Atest = {}
        
        self.Aval_run = {}
        self.Atest_run = {}
        
        # standard error of mean
        self.Ssize_task = {}
        self.Sforget_task = {}
        self.Sval_task = {}
        self.Stest_task = {}
        
        self.Ssize = {}
        self.Sforget = {}
        self.Sval = {}
        self.Stest = {}
        
        self.Sval_run = {}
        self.Stest_run = {}
        
        for l1_hyp in self.l1_hyp_list:
            self.Asize_task[l1_hyp] = {}
            self.Aforget_task[l1_hyp] = {}
            self.Aval_task[l1_hyp] = {}
            self.Atest_task[l1_hyp] = {}
            
            self.Asize[l1_hyp] = {}
            self.Aforget[l1_hyp] = {}
            self.Aval[l1_hyp] = {}
            self.Atest[l1_hyp] = {}
            
            self.Aval_run[l1_hyp] = {}
            self.Atest_run[l1_hyp] = {}
            
            self.Ssize_task[l1_hyp] = {}
            self.Sforget_task[l1_hyp] = {}
            self.Sval_task[l1_hyp] = {}
            self.Stest_task[l1_hyp] = {}
            
            self.Ssize[l1_hyp] = {}
            self.Sforget[l1_hyp] = {}
            self.Sval[l1_hyp] = {}
            self.Stest[l1_hyp] = {}
            
            self.Sval_run[l1_hyp] = {}
            self.Stest_run[l1_hyp] = {}
            for reg_coef in self.reg_coef_list:
                self.Asize_task[l1_hyp][reg_coef] = self.tables['increase'][l1_hyp][reg_coef].mean(0).round(3)
                self.Aforget_task[l1_hyp][reg_coef] = self.forget[l1_hyp][reg_coef].mean(0).round(2)
                self.Aval_task[l1_hyp][reg_coef] = self.tables['val'][l1_hyp][reg_coef][:,n_task-1,:].mean(0).round(2)
                self.Atest_task[l1_hyp][reg_coef] = self.tables['test'][l1_hyp][reg_coef][:,n_task-1,:].mean(0).round(2)
                
                self.Asize[l1_hyp][reg_coef] = self.tables['increase'][l1_hyp][reg_coef][:,n_task-1].mean().round(3)
                self.Aforget[l1_hyp][reg_coef] = self.forget[l1_hyp][reg_coef].mean().round(2)
                self.Aval[l1_hyp][reg_coef] = self.tables['val'][l1_hyp][reg_coef][:,n_task-1,:].mean().round(2)
                self.Atest[l1_hyp][reg_coef] = self.tables['test'][l1_hyp][reg_coef][:,n_task-1,:].mean().round(2)
                
                self.Aval_run[l1_hyp][reg_coef] = self.tables['val'][l1_hyp][reg_coef][:,n_task-1,:].mean(1).round(2)
                self.Atest_run[l1_hyp][reg_coef] = self.tables['test'][l1_hyp][reg_coef][:,n_task-1,:].mean(1).round(2) 
                
                self.Ssize_task[l1_hyp][reg_coef] = (self.tables['increase'][l1_hyp][reg_coef].std(0)/sqrt(repeat)).round(3)
                self.Sforget_task[l1_hyp][reg_coef] = (self.forget[l1_hyp][reg_coef].std(0)/sqrt(repeat)).round(2)
                self.Sval_task[l1_hyp][reg_coef] = (self.tables['val'][l1_hyp][reg_coef][:,n_task-1,:].std(0)/sqrt(repeat)).round(2)
                self.Stest_task[l1_hyp][reg_coef] = (self.tables['test'][l1_hyp][reg_coef][:,n_task-1,:].std(0)/sqrt(repeat)).round(2) 
                
                self.Ssize[l1_hyp][reg_coef] = self.Ssize_task[l1_hyp][reg_coef][n_task-1]
                self.Sforget[l1_hyp][reg_coef] = (self.forget[l1_hyp][reg_coef].mean(1).std()/sqrt(repeat)).round(2)
                self.Sval[l1_hyp][reg_coef] = (self.tables['val'][l1_hyp][reg_coef][:,n_task-1,:].mean(1).std()/sqrt(repeat)).round(2)
                self.Stest[l1_hyp][reg_coef] = (self.tables['test'][l1_hyp][reg_coef][:,n_task-1,:].mean(1).std()/sqrt(repeat)).round(2)
                
                self.Sval_run[l1_hyp][reg_coef] = (self.tables['val'][l1_hyp][reg_coef][:,n_task-1,:].std(1)/sqrt(n_task)).round(2)
                self.Stest_run[l1_hyp][reg_coef] = (self.tables['test'][l1_hyp][reg_coef][:,n_task-1,:].std(1)/sqrt(n_task)).round(2) 

    def getOptimal(self):
        # Return optimal regularization values based on average validation error
        self.optim_l1_hyp = []
        self.optim_reg_coef = []
        
        self.optim_Asize = 0.0
        self.optim_Aforget = 0.0
        self.optim_Aval = 0.00
        self.optim_Atest = 0.00
        
        self.optim_Ssize = 0.0
        self.optim_Sforget = 0.0
        self.optim_Sval = 0.00
        self.optim_Stest = 0.00
        i = 0
        
        for l1_hyp in self.l1_hyp_list:
            for reg_coef in self.reg_coef_list:
                if self.Aval[l1_hyp][reg_coef] > self.optim_Aval:
                    self.optim_Aval = self.Aval[l1_hyp][reg_coef] 
                    self.optim_Sval = self.Sval[l1_hyp][reg_coef]
                    
        for l1_hyp in self.l1_hyp_list:
            for reg_coef in self.reg_coef_list:
                if self.Aval[l1_hyp][reg_coef] == self.optim_Aval:
                    self.optim_l1_hyp.append(l1_hyp)
                    self.optim_reg_coef.append(reg_coef)
                    if(i==0):
                        self.optim_Asize = self.Asize[l1_hyp][reg_coef]
                        self.optim_Aforget = self.Aforget[l1_hyp][reg_coef]
                        self.optim_Atest = self.Atest[l1_hyp][reg_coef] 
                        
                        self.optim_Asize_task = self.Asize_task[l1_hyp][reg_coef]
                        self.optim_Aforget_task = self.Aforget_task[l1_hyp][reg_coef]
                        self.optim_Aval_task = self.Aval_task[l1_hyp][reg_coef]
                        self.optim_Atest_task = self.Atest_task[l1_hyp][reg_coef]
                        self.optim_Aval_run = self.Aval_run[l1_hyp][reg_coef]
                        self.optim_Atest_run = self.Atest_run[l1_hyp][reg_coef]
                        
                        self.optim_Ssize = self.Ssize[l1_hyp][reg_coef]
                        self.optim_Sforget = self.Sforget[l1_hyp][reg_coef]
                        self.optim_Stest = self.Stest[l1_hyp][reg_coef]   
                        
                        self.optim_Ssize_task = self.Ssize_task[l1_hyp][reg_coef]
                        self.optim_Sforget_task = self.Sforget_task[l1_hyp][reg_coef]
                        self.optim_Sval_task = self.Sval_task[l1_hyp][reg_coef]
                        self.optim_Stest_task = self.Stest_task[l1_hyp][reg_coef]
                        self.optim_Sval_run = self.Sval_run[l1_hyp][reg_coef]
                        self.optim_Stest_run = self.Stest_run[l1_hyp][reg_coef]
                        
                    i += 1
                    

