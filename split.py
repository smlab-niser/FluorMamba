import json
import numpy as np

filename = 'osfp'

L = np.array(json.load(open(f'datasets/{filename}.json','r')))
np.random.seed(0)
np.random.shuffle(L)

lists = np.array_split(L, 10)
for i, l in enumerate(lists):
    with open(f'datasets/{filename}_{i}.json','w') as f:
        f.write(json.dumps(l.tolist()))