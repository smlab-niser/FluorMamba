import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import balanced_accuracy_score as baccuracy
from sklearn.metrics import f1_score as f1
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as preci


def set_cuda(num):
    device = torch.device(f"cuda:{num}" if num<4 else "cpu")
    print(f'\nDevice set to {device.type}:{num}')
    return device

def replace(L, p=10):
    map = {'A':'V', 'S':'T', 'F':'Y', 'K':'R', 'C':'M', 'D':'E', 'N':'Q', 'V':'I'}
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        flag=0
        l=k
        for i,aa in enumerate(k):
            r = np.random.randint(100)
            if r<p and aa in map:
                flag=1
                l = l[:i] + map[aa] + l[i+1:]
        if flag:
           M.append((l,v))
    np.random.shuffle(M)
    return M

def reverse(L):
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        l = k[::-1]
        M.append((l,v))
    np.random.shuffle(M)
    return M

def repeat_expansion(L, p=10):
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        flag=0
        l=k
        for i,_ in enumerate(k):
            r = np.random.randint(100)
            if r<p:
                flag=1
                s = np.random.randint(1,5)
                l = l[:i] + l[i:i+s]*2 + l[i+s:]
        if flag:
           M.append((l,v))
    np.random.shuffle(M)
    return M    

def subsample(L):
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        alpha = np.random.randint(0,len(k)-1)
        l = k[alpha:min(alpha+50,len(k))]
        M.append((l,v))
    np.random.shuffle(M)
    return M

def subshuffle(L):
    import random
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        alpha = np.random.randint(0,len(k)-1)
        l = k[alpha:min(alpha+np.random.randint(1,20),len(k))]
        l_ = ''.join(random.sample(l,len(l)))
        k_ = k[:alpha] + l_ + k[alpha+len(l):]
        M.append((k_,v))
    np.random.shuffle(M)
    return M

def swap(L, p=10):
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        flag=0
        l=list(k)
        for i,aa in enumerate(k):
            r = np.random.randint(100)
            if r<p:
                flag=1
                s = np.random.randint(len(l))
                l[i],l[s] = l[s],l[i]
        if flag:
           M.append((''.join(l),v))
    np.random.shuffle(M)
    return M

def delete(L, p=10):
    M=[]
    for k,v in tqdm(L, desc='Processing', total=len(L)):
        flag=0
        l=k
        for i,_ in enumerate(k):
            r = np.random.randint(100)
            if r<p:
                flag=1
                l = l[:i] + l[i+1:]
        if flag:
           M.append((l,v))
    np.random.shuffle(M)
    return M

def load_oligo_database(filename,seed):
    file = open('datasets/'+filename,'r')
    l = json.load(file)
    file.close()
    np.random.seed(seed)
    print("seed=",seed)
    np.random.shuffle(l)
    keys = []
    values = []
    for i in l:
        keys.append(i[0])
        values.append(float(i[1]))
    return keys, F.one_hot(torch.tensor(values).type(torch.int64), num_classes=2).float().tolist()

def load_oligo_database2(filename,seed):
    file = open('datasets/'+filename,'r')
    l = json.load(file)
    file.close()
    np.random.seed(seed)
    np.random.shuffle(l)
    a1=replace(l, p=10)
    a2=reverse(l)
    a3=subsample(l)
    a4=repeat_expansion(l)
    a5=subshuffle(l)
    a6=swap(l)
    a7=delete(l, p=10)
    keys = []
    values = []
    l=l+a1+a2
    for i in l:
        keys.append(i[0])
        values.append(float(i[1]))
    return keys, F.one_hot(torch.tensor(values).type(torch.int64), num_classes=2).float().tolist()

def load_multiple_oligo_database(filenames,seed):
    keys=[]
    values=[]
    for filename in filenames:
        key, value = load_oligo_database(filename,seed)
        keys += key
        values += value
    return keys, values

def load_multiple_oligo_database2(filenames,seed):
    keys=[]
    values=[]
    for filename in filenames:
        key, value = load_oligo_database2(filename,seed)
        keys += key
        values += value
    return keys, values

class OligoDataset(Dataset):
    def __init__(self, x, y, device):
        self.x = x
        self.y = y
        self.device = device

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]).to(self.device), torch.tensor(self.y[idx]).to(self.device)

class Tokenizer():
    def __init__(self, padding=False):
        self.vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
        self.pad_token_id = 20
        self.padding = padding
    def __call__(self, seq, max_length=467):
        if self.padding:
            return [self.vocab.index(c) for c in seq]+[self.pad_token_id]*(max_length-len(seq))
        return [self.vocab.index(c) for c in seq]

def get_dataloader(train_filenames, test_filename, batch_size, device, padding=False, tokenize='learned', max_length=467,seed=0):
    np.random.seed(seed)
    print('\n Train Data Loading...')
    train_sequences, train_labels = load_multiple_oligo_database2(train_filenames,seed)
    print('train Data Loaded\n')
    print('\n Test Data Loading...')
    if isinstance(test_filename, str):
        test_sequences, test_labels = load_oligo_database(test_filename,seed)  
    else:
        test_sequences, test_labels = [],[]
    print('Test Data Loaded\n')
    sequences, labels = train_sequences+test_sequences, train_labels+test_labels    
    if tokenize == 'learned':
        tok = Tokenizer(padding)
        tok_sequences = [tok(seq, max_length) for seq in tqdm(sequences, total=len(sequences), desc='Tokenizing')]
        print('\n Sequences Tokenized')
    elif tokenize == 'glove':
        tok_sequences = [(seq+[[0.0]*512]*(467 - len(seq))) for seq in sequences]
    elif tokenize == 'protvec':
        tok_sequences = sequences
    if isinstance(test_filename, str):
        num = -1*(len(test_sequences))
    else:
        num = -1*test_filename
    train_dataset = OligoDataset(tok_sequences[:num], labels[:num], device)
    test_dataset = OligoDataset(tok_sequences[num:], labels[num:], device)
    print('Training dataset size: ', len(train_dataset))
    print('Test dataset size: ', len(test_dataset))
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return dataloader, test_dataloader

def train(model, input_tensor, category_tensor, criterion, optimizer):
    model.train()
    output = model(input_tensor)
    loss = criterion(output, category_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    
    return output.argmax(dim=1).tolist(), loss.item()

def evaluate(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        return output.argmax(dim=1).item()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f'\nModel saved as {filename}\n')

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    print(f'\nModel loaded from {filename}\n')
    return model

def train_loop(model, dataloader, test_dataloader, filename, learning_rate = 1e-4, wd=0.01,  n_iters=500, seed=0, plot=True):
    time = list(datetime.now().timetuple())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    torch.manual_seed(seed)
    epochs=[]
    all_losses = []
    all_train_accuracies = []
    all_accuracies = []
    all_balanced_accuracies = []
    all_f1_scores = []
    all_recall = []
    all_precision = []
    all_mcc = []
    print(f'\nTraining {model.__name__} on {filename}...')
    for epoch in tqdm(range(n_iters), desc='Epochs'):
        losses=[]
        train_outputs = []
        ys = []
        for _, (x,y) in enumerate(dataloader):#tqdm(enumerate(dataloader), desc='Batches', total=len(dataloader)):
            train_out, loss = train(model, x, y, criterion, optimizer)
            losses.append(loss)
            train_outputs+=train_out
            ys+=y.argmax(dim=1).tolist()
        epochs.append(epoch)
        all_losses.append(np.mean(losses))
        train_acc = accuracy(train_outputs, ys)
        all_train_accuracies.append(train_acc)
        if plot:
            plt.plot(epochs, all_losses)
            plt.savefig(f'results/{model.__name__}_loss_{filename}.png')
            plt.close()

        test_outputs = []
        targets = []
        for _, (x,y) in enumerate(test_dataloader):
            test_outputs.append(evaluate(model, x))
            targets.append(y.argmax().item())
        acc = accuracy(targets, test_outputs)
        b_acc = baccuracy(targets, test_outputs)
        f1_score = f1(targets, test_outputs)
        mcc_score = mcc(targets, test_outputs)
        recall_score = recall(targets, test_outputs)
        percision_score = preci(targets, test_outputs)
        all_accuracies.append(acc)
        all_balanced_accuracies.append(b_acc)
        all_f1_scores.append(f1_score)
        all_mcc.append(mcc_score)
        all_recall.append(recall_score)
        all_precision.append(percision_score)
        if plot:
            plt.plot(epochs, all_train_accuracies)
            plt.plot(epochs, all_accuracies)
            plt.plot(epochs, all_balanced_accuracies)
            plt.plot(epochs, all_f1_scores)
            plt.plot(epochs, all_mcc)
            plt.plot(epochs, all_recall)
            plt.plot(epochs, all_precision)
            plt.legend(['Train Accuracy', 'Accuracy', 'Balanced Accuracy', 'F1 Score', 'MCC', 'Recall', 'Precision'])
            plt.savefig(f'results/{model.__name__}_accuracy_{filename}.png')
            plt.close()
        print(f'\nLoss : {np.mean(losses):.4f}  Train_Accuracy : {train_acc:.4f}  Accuracy : {acc:.4f}  Balanced Accuracy : {b_acc:.4f}  F1 Score : {f1_score:.4f}  MCC : {mcc_score:.4f} Recall : {recall_score:.4f}  Precision : {percision_score:.4f}\n')

    return model, [all_accuracies, all_balanced_accuracies, all_f1_scores, all_mcc, all_recall, all_precision]