from utils_aug import set_cuda, get_dataloader, train_loop, save_model, json, np
from models import MambaTower, Transformer, RNN, LSTM


number = 0

def main(number, flag ,model,dataset, cuda_number=2):

    device = set_cuda(cuda_number)
    filename = dataset
    if(flag=="10CV"):
        test = f'{filename}_{number}.json'  # Change this variable to update the test file name
        train_filename = [f'{filename}_{i}.json' for i in range(10) if f'{filename}_{i}.json' != test]
        out_filename = f'{filename}__{test.split("_")[1]}_exp'
    elif(flag=="100ID"):
        train_filename = [f'{filename}.json']
        test=70
        l = json.load(open('datasets/'+train_filename[0],'r'))
        np.random.seed(number)
        np.random.shuffle(l)
        with open('datasets/100ID_train.json','w') as f:
            f.write(json.dumps(l[:-test]))
        with open('datasets/100ID_test.json','w') as f:
            f.write(json.dumps(l[-test:]))
        train_filename = ['100ID_train.json']
        test = '100ID_test.json'
    
    model_name = model

    embedding = 'learned'
    batch_size = 8
    n_layers = 2
    dropout = 0.3
    dim_model = 512
    state_size = 64
    d_conv = 3
    expand = 2
    n_head = 8
    learning_rate = 1e-5

    save_the_file = True

    seq_len = 467 if train_filename == 'osfp.json' else 582
    
    if model_name == 'MambaTower':
        model = MambaTower(dim_model, n_layers, state_size, dropout, global_pool=True, d_conv=d_conv, expand=expand, embedding=embedding, device=device).to(device)
    if model_name == 'Transformer':
        model = Transformer(dim_model,n_head ,n_layers, state_size, dropout, global_pool=True, expand=expand).to(device)
    if model_name == 'RNN':
        model = RNN(dim_model, state_size, n_layers, dropout, global_pool=True, expand=expand).to(device)
    if model_name == 'LSTM':
        model = LSTM(dim_model, state_size, n_layers, dropout, global_pool=True, expand=expand).to(device)
    
    print(f'\nModel : {model.__name__}\nDataset : {train_filename}\n')
    out_filename=model.__name__+str(number)
    train_data, test_data = get_dataloader(train_filename, test, batch_size, device, padding=True, tokenize=embedding, max_length=seq_len,seed=number)
    model,evaln = train_loop(model, train_data, test_data, out_filename, learning_rate = learning_rate, wd=0.01,  n_iters=100, seed=number, plot=True)

    if save_the_file:
        save_model(model, f'models/{model.__name__}_{out_filename}.pt')
    print(evaln)

    return evaln

if __name__ == '__main__':
    main(number, '10CV', 'MambaTower', 'osfp')