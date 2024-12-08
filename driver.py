from main import main
import json
from tqdm import tqdm


L=[]
flag = "100ID"
model = "MambaTower"
dataset="osfp"
if (flag=="10CV"):
    for number in tqdm(range(10)):
        L.append(main(number,flag,model,dataset))
elif(flag=="100ID"):
    for number in tqdm(range(100)):
        L.append(main(number,flag,model,dataset))


with open('results_rnn_10x.json','w') as f:
    json.dump(L,f)

