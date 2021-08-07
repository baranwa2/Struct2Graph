# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:36:25 2019

@author: mayank
"""

import pickle
import timeit
import os
import random
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import resample

from sklearn.metrics import roc_auc_score
from sklearn.metrics.ranking import _binary_clf_curve

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_trials = 5

class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_gnn             = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W1_attention      = nn.Linear(dim, dim)
        self.W2_attention      = nn.Linear(dim, dim)
        self.w                 = nn.Parameter(torch.zeros(dim))
        
        self.W_out             = nn.Linear(2*dim, 2)
        
    def gnn(self, xs1, A1, xs2, A2):
        for i in range(layer_gnn):
            hs1 = torch.relu(self.W_gnn[i](xs1))            
            hs2 = torch.relu(self.W_gnn[i](xs2))
            
            xs1 = torch.matmul(A1, hs1)
            xs2 = torch.matmul(A2, hs2)
        
        return xs1, xs2
        
    
    def mutual_attention(self, h1, h2):
        x1 = self.W1_attention(h1)
        x2 = self.W2_attention(h2)
        
        m1 = x1.size()[0]
        m2 = x2.size()[0]
        
        c1 = x1.repeat(1,m2).view(m1, m2, dim)
        c2 = x2.repeat(m1,1).view(m1, m2, dim)

        d = torch.tanh(c1 + c2)
        alpha = torch.matmul(d,self.w).view(m1,m2)
        
        b1 = torch.mean(alpha,1)
        p1 = torch.softmax(b1,0)
        s1 = torch.matmul(torch.t(x1),p1).view(-1,1)
        
        b2 = torch.mean(alpha,0)
        p2 = torch.softmax(b2,0)
        s2 = torch.matmul(torch.t(x2),p2).view(-1,1)
        
        return torch.cat((s1,s2),0).view(1,-1), p1, p2
    
    def forward(self, inputs):

        fingerprints1, adjacency1, fingerprints2, adjacency2 = inputs
        
        """Protein vector with GNN."""
        x_fingerprints1        = self.embed_fingerprint(fingerprints1)
        x_fingerprints2        = self.embed_fingerprint(fingerprints2)
        
        x_protein1, x_protein2 = self.gnn(x_fingerprints1, adjacency1, x_fingerprints2, adjacency2)
        
        """Protein vector with mutual-attention."""
        y, p1, p2     = self.mutual_attention(x_protein1, x_protein2)
        z_interaction = self.W_out(y)

        return z_interaction, p1, p2
    
    def __call__(self, data, train=True):
        
        inputs, t_interaction = data[:-1], data[-1]
        z_interaction, p1, p2 = self.forward(inputs)
        
        if train:
            loss = F.cross_entropy(z_interaction, t_interaction)
            return loss
        else:
            z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
            t = int(t_interaction.to('cpu').data[0].numpy())
            return z, t, p1, p2


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        
        sampling  = random.choices(dataset, k=800)
        
        loss_total = 0
        for data in sampling:
            
            s1, i1 = file_ind(data[0])
            s2, i2 = file_ind(data[1])
            
            A1 = np.load(dir_input+'adjacencies_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            A2 = np.load(dir_input+'adjacencies_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)
            
            P1 = np.load(dir_input+'proteins_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            P2 = np.load(dir_input+'proteins_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)
            
            protein1 = torch.LongTensor(P1[i1])
            protein2 = torch.LongTensor(P2[i2])
            
            adjacency1 = torch.FloatTensor(A1[i1])
            adjacency2 = torch.FloatTensor(A2[i2])
            
            interaction = torch.LongTensor([data[2]])
            
            comb = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device), interaction.to(device))
            
            loss = self.model(comb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):

        #sampling  = random.choices(dataset, k=200)
        sampling = dataset
        
        z_list, t_list = [], []
        for data in sampling:
            
            s1, i1 = file_ind(data[0])
            s2, i2 = file_ind(data[1])
            
            A1 = np.load(dir_input+'adjacencies_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            A2 = np.load(dir_input+'adjacencies_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)
            
            P1 = np.load(dir_input+'proteins_'+ str(s1+1) + '_' + str(s1+10) + '.npy', allow_pickle=True)
            P2 = np.load(dir_input+'proteins_'+ str(s2+1) + '_' + str(s2+10) + '.npy', allow_pickle=True)
            
            protein1 = torch.LongTensor(P1[i1])
            protein2 = torch.LongTensor(P2[i2])
            
            adjacency1 = torch.FloatTensor(A1[i1])
            adjacency2 = torch.FloatTensor(A2[i2])
            
            interaction = torch.LongTensor([data[2]])
            
            comb = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device), interaction.to(device))
            
            z, t, _, _ = self.model(comb, train=False)
            z_list.append(z)
            t_list.append(t)

        score_list, label_list = [], []
        for z in z_list:
            score_list.append(z[1])
            label_list.append(np.argmax(z))

        labels = np.array(label_list)
        y_true = np.array(t_list)
        y_pred = np.array(score_list)

        tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv = calculate_performace(len(sampling), labels,  y_true)
        roc_auc_val = roc_auc_score(t_list, score_list)
        fpr, tpr, thresholds = roc_curve(labels, y_pred)
        auc_val = auc(fpr, tpr)

        return accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn, y_true,  y_pred

    def result(self, epoch, time, loss, accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn, file_name):
        with open(file_name, 'a') as f:
            result = map(str, [epoch, time, loss, accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn])
            f.write('\t'.join(result) + '\n')

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)
                
    def eval_PSM(self, z_list, file_name):
        with open(file_name, 'a') as f:
            f.write(str(z_list) + '\n')

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1        
                
                
    if (tp+fn) == 0:
        q9 = float(tn-fp)/(tn+fp + 1e-06)
    if (tn+fp) == 0:
        q9 = float(tp-fn)/(tp+fn + 1e-06)
    if  (tp+fn) != 0 and (tn+fp) !=0:
        q9 = 1- float(np.sqrt(2))*np.sqrt(float(fn*fn)/((tp+fn)*(tp+fn))+float(fp*fp)/((tn+fp)*(tn+fp)))
        
    Q9 = (float)(1+q9)/2
    accuracy = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    recall = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    ppv = float(tp)/(tp + fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    F1_score = float(2*tp)/(2*tp + fp + fn + 1e-06)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    
    
    return tp,fp,tn,fn,accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def file_ind(index):
    st_ind, in_ind = divmod(index,10)
    return 10*st_ind, in_ind


radius         = 1
dim            = 20
layer_gnn      = 2
lr             = 1e-3
lr_decay       = 0.5
decay_interval = 10
iteration      = 100

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

dir_input    = ('pdb_files/input'+str(radius)+'/')
examples         = np.load(dir_input + 'train_examples_one_ten.npy')
fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
n_fingerprint = len(fingerprint_dict) + 100

#p_names = ['5KHB', '6NIV', '5KGZ', '2LY4']
p_names = ['1R19']
p_list = {}
A_list = {}

filenames = glob.glob(dir_input + 'names*.npy')
for f in filenames:
    n = np.load(f, allow_pickle=True)
    
    if '1R17' in n:
        ind = f.strip().split('s')[2]
        ind_1R17 = np.where(n=='1R17')[0]
        A_1R17   = np.load(dir_input + 'adjacencies'+str(ind), allow_pickle=True)[ind_1R17][0]
        p_1R17   = np.load(dir_input + 'proteins'+str(ind), allow_pickle=True)[ind_1R17][0]
        
        p_list['1R17'] = p_1R17
        A_list['1R17'] = A_1R17


    if '1R19' in n:
        ind = f.strip().split('s')[2]
        ind_1R19 = np.where(n=='1R19')[0]
        A_1R19   = np.load(dir_input + 'adjacencies'+str(ind), allow_pickle=True)[ind_1R19][0]
        p_1R19   = np.load(dir_input + 'proteins'+str(ind), allow_pickle=True)[ind_1R19][0]
        
        p_list['1R19'] = p_1R19
        A_list['1R19'] = A_1R19

n_samples = np.ceil(0.8*len(examples)).astype(int)

for trial in range(num_trials):
    #Bootstrap resampling
    dataset_train = resample(examples, replace=True, n_samples=n_samples, random_state=trial)
    dataset_test  = np.array([x for x in examples.tolist() if x not in dataset_train.tolist()])
    
    torch.manual_seed(1234)
    model   = ProteinProteinInteractionPrediction().to(device)
    trainer = Trainer(model)
    tester  = Tester(model)

    file_result = 'output/result/ten/boot/' + 'results_trial_' + str(trial+1) + '.txt'
    os.makedirs('output/result/ten/boot/', exist_ok=True)
    with open(file_result, 'w') as f:
        f.write('Epoch \t Time(sec) \t Loss_train \t Accuracy \t Precision \t Recall \t Sensitivity \t Specificity \t MCC \t F1-score \t ROC_AUC \t AUC \t Q9 \t PPV \t NPV \t TP \t FP \t TN \t FN\n')

    file_model = 'output/model/ten/boot/' + 'model_trial_' + str(trial+1)
    os.makedirs('output/model/ten/boot/', exist_ok=True)
    
    file_eval   = 'output/result/ten/boot/' + 'evaluations_trial_' + str(trial+1) + '.txt'
    
    file_p1     = 'output/result/ten/boot/' + 'p1_trial_' + str(trial+1) + '.txt'
    file_p2     = 'output/result/ten/boot/' + 'p2_trial_' + str(trial+1) + '.txt'
    with open(file_p1, 'w') as f:
        f.write('')
    with open(file_p2, 'w') as f:
        f.write('')
    with open(file_eval, 'w') as f:
        f.write('')

    print('Training...')
    start = timeit.default_timer()
    
    for epoch in range(iteration):
        if (epoch+1) % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss = trainer.train(dataset_train)
        accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn, y_true, y_pred = tester.test(dataset_test)
        fpr, tpr, thresh = roc_curve(y_true, y_pred, pos_label=1)
        lr_fps, lr_tps, _ = _binary_clf_curve(y_true, y_pred)
        lr_recall = lr_tps / lr_tps[-1]
        lr_specificity = (lr_fps[-1] - lr_fps) / lr_fps[-1]
        last_ind = lr_tps.searchsorted(lr_tps[-1])
        sl = slice(last_ind, None, -1)
        lr_specificity = np.r_[lr_specificity[sl], 1]
        lr_recall = np.r_[lr_recall[sl], 0]
        lr_precision = lr_recall/(lr_recall + 100*(1-lr_specificity) + 1e-06) #Prevalence corrected precision
        
        end  = timeit.default_timer()
        time = end - start

        tester.result(epoch, time, loss,  accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, auc_val, Q9, ppv, npv, tp, fp, tn, fn, file_result)
        tester.save_model(model, file_model)

        print('Epoch: ' + str(epoch))
        print('Accuracy: ' + str(accuracy))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Sensitivity: ' + str(sensitivity))
        print('Specificity: ' + str(specificity))
        print('MCC: ' + str(MCC))
        print('F1-score: ' + str(F1_score))
        print('ROC-AUC: ' + str(roc_auc_val))
        print('AUC: ' + str(auc_val))
        print('Q9: ' + str(Q9))
        print('PPV: ' + str(ppv))
        print('NPV: ' + str(npv))
        print('TP: ' + str(tp))
        print('FP: ' + str(fp))
        print('TN: ' + str(tn))
        print('FN: ' + str(fn))
        print('\n')
        
        plt.figure()
        plt.plot(lr_recall[:-1], lr_precision[:-1], linestyle='--', linewidth=2, color='blue', label='Struct2Graph')
        plt.title('Precision-Recall curve: Unbalanced dataset (1:10)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='best')
        plt.savefig('Figures/one-ten/boot/' + 'PR_epoch_'+str(epoch+1)+'_trial_'+str(trial+1)+'_unbalanced'+'.svg',dpi=300)
        plt.savefig('Figures/one-ten/boot/' + 'PR_epoch_'+str(epoch+1)+'_trial_'+str(trial+1)+'_unbalanced'+'.png',dpi=300)
        plt.close('all')


        z_list = []
        p1_list, p2_list = [], []
        p1 = '1R17'
        protein1   = torch.LongTensor(p_list[p1])
        adjacency1 = torch.FloatTensor(A_list[p1])
        for p2 in p_names:
            protein2    = torch.LongTensor(p_list[p2])
            adjacency2  = torch.FloatTensor(A_list[p2])
            
            interaction = torch.LongTensor([0])
            comb        = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device), interaction.to(device))
            z,_,pr1,pr2 = model(comb, train=False)
            
            z_list.append(z)
            p1_list.append(list(pr1.data.cpu().numpy()))
            p2_list.append(list(pr2.data.cpu().numpy()))
                
        tester.eval_PSM(z_list, file_eval)
        tester.eval_PSM(p1_list, file_p1)
        tester.eval_PSM(p2_list, file_p2)