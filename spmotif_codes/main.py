
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os.path as osp
import random
## dataset
from sklearn.model_selection import train_test_split
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from spmotif_dataset import SPMotif

import random
## training
from model import EaDA
from utils import init_weights, get_args, eval_spmotif, train_eada, get_kmeans
import copy
import json
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

args_first = get_args()
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.DEBUG,  #INFO,
                    )
logger = logging.getLogger(__name__)

def main(args,seed):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    datadir = './data/'
    bias = args.bias
    train_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='train')
    labels = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='train').y
    val_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='test')
    
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ## split the client
    random.seed(42)
    np.random.seed(42)
    def partition_labels_with_dirichlet_distribution(N, alpha, client_num, idx_batch, labels_batch):
        np.random.shuffle(labels_batch)  

        proportions = np.random.dirichlet(np.repeat(alpha, client_num))
        
        print("Generated proportions (per client):")
        print(proportions)
        weight = proportions

        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(labels_batch)).astype(int)[:-1]
        
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(labels_batch, proportions))]
        
        min_size = min([len(idx_j) for idx_j in idx_batch]) 

        return idx_batch, min_size, weight

    def create_non_uniform_split(alpha, idxs, labels, client_number):
        N = len(idxs)
        print(f"Total number of samples: {N}, Total clients: {client_number}")
        idx_batch_per_client = [[] for _ in range(client_number)] 
        idxs_per_label = [np.where(labels == i)[0] for i in range(3)]  

        idx_batch_per_client, min_size, weight = partition_labels_with_dirichlet_distribution(
            N, alpha, client_number, idx_batch_per_client, idxs
        )
        
        return idx_batch_per_client, weight

    
    def get_fed_dataset(train_data, labels, client_number, alpha ):
        num_train_samples = len(train_data)
        train_idxs = list(range(num_train_samples))
        random.shuffle(train_idxs)

        clients_idxs_train,weight = create_non_uniform_split(
                alpha, train_idxs, labels, client_number
            )
        # print(clients_idxs_train)
        partition_dicts = [None] * client_number

        for client in range(client_number):
            client_train_idxs = clients_idxs_train[client]

            train_client = [
                train_data[idx] for idx in client_train_idxs
            ]

            partition_dict = {
            "train": train_client,
            }

            partition_dicts[client] = partition_dict

        return partition_dicts,weight


    partition_dicts_clients,client_weights = get_fed_dataset(train_dataset, labels, client_number=3, alpha=args.alpha)

    #  将各个客户端的数据封装为dataloader
    partition_dicts = [None] * args.client_number
    for client in range(args.client_number):
        train_loader = DataLoader(partition_dicts_clients[client]['train'], batch_size=args.batch_size, shuffle=True, num_workers = 0)
        partition_dict = {
        "train": train_loader,
        }
        partition_dicts[client] = partition_dict
        print(len(train_loader))

    set_seed(seed)

    # n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), float(len(test_loader.dataset))
    # logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")


    server_model = eval(args.model_name)( gnn_type = args.gnn, num_tasks = 3, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device)   
                          
    init_weights(server_model, args.initw_name, init_gain=0.02)

    models = [copy.deepcopy(server_model) for idx in range(args.client_number)]


    def get_opt(args,model):
    
        opt_separator = optim.Adam(list(model.separator.parameters())  + list(model.node_enoder.parameters()), lr=args.lr, weight_decay=args.l2reg)
        opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters())  + list(model.node_enoder.parameters()), lr=args.lr, weight_decay=args.l2reg)

        optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
        if args.use_lr_scheduler:
            schedulers = {}
            for opt_name, opt in optimizers.items():
                schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
        else:
            schedulers = None
        return optimizers,schedulers

    def communication(server_model, models, client_weights):
        client_num = len(models)
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                # 参数聚合
                # print(client_num)
                for client_idx in range(client_num):
                    #  print(client_weights[client_idx])
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                # 参数分发
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        return server_model, models

    cnt_wait = 0
    best_epoch = 0
    loss_logger = []
    valid_logger = []
    test_logger = []
    client_test_all = [[],[],[],[]]
    client_valid_all = [[],[],[],[]]
    task_type = ["classification"]
    communication_env_all = None
    for epoch in range(args.epochs):
        
        all_opt_and_sch = [ get_opt(args, models[idx])  for idx in range(args.client_number)]
        
    
        for client_idx, model in enumerate(models):
            for i in range(20):
                path = i % int(args.path_list[-1])
                if path in list(range(int(args.path_list[0]))):
                    optimizer_name = 'separator' 
        
                elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
                    optimizer_name = 'predictor'


                model.train()

                optimizers,schedulers = all_opt_and_sch[client_idx]
                train_loader = partition_dicts[client_idx]['train']
                
                train_eada(args, model, device, train_loader, optimizers, task_type, optimizer_name,loss_logger,communication_env_all,server_model)
                
                if schedulers != None:
                    schedulers[optimizer_name].step()

        with torch.no_grad():
            bias_models = [copy.deepcopy(bias_model) for bias_model in models]
            communication_env = []
            for client_idx,env_model in enumerate(models):
                _, cluster_centers,_ = get_kmeans(args, env_model, device, partition_dicts[client_idx]['train'])
                communication_env.append(cluster_centers)
            communication_env_all = torch.cat(communication_env,0)
            communication_env_all = communication_env_all.to(device)  
            

            server_model, models = communication( server_model, models, client_weights)


        server_model.eval()
        valid_perf = eval_spmotif(args, server_model, device, valid_loader)[0]
        
        valid_logger.append(valid_perf)
        
        update_test = False

        test_perfs = eval_spmotif(args, server_model, device, test_loader)
        # class_test_perfs = eval_spmotif_class(args, server_model, device, test_loader, evaluator)
        test_auc  = test_perfs[0]
        # class_test_auc = class_test_perfs[0]
        print("=====Epoch {}, Metric: {}, Validation: {}, Test: {}, Class_Test:{}".format(epoch, 'AUC', valid_perf, test_auc,0))

        if epoch != 0:
            if 'classification' in task_type and valid_perf >  best_valid_perf:
                update_test = True
            elif 'classification' not in task_type and valid_perf <  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            test_auc1 = test_auc
            # cnt_wait = 0
            best_epoch = epoch
        else:
            # print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break

    logger.info('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))
    print('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))


    
    logger.info('Test auc: {}'.format(test_auc))
    return [best_valid_perf, test_auc1]

    

    

def config_and_run(args):
    
    if args.by_default:
        if args.dataset == 'spmotif':
            args.epochs = 10
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
                args.l2reg = 1e-3
                args.gamma = 0.55
                args.num_layer = 2  
                args.batch_size = 32
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 40
                args.drop_ratio = 0.3
                args.initw_name = 'orthogonal' 
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
                args.patience = 40
                args.initw_name = 'orthogonal' 
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 32

    for k, v in vars(args).items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    args.plym_prop = 'none' 
    
    results = {'valid_auc': [], 'test_auc': []}
    
    for seed in range(args.trails):

        set_seed(seed)
        valid_auc, test_auc = main(args,seed)
        results['valid_auc'].append(valid_auc)
        results['test_auc'].append(test_auc)

    for mode, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))

        print('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))
        xx = '{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums)
        json_str = json.dumps({'result': xx }, ensure_ascii=False)

if __name__ == "__main__":
    args = get_args()
    config_and_run(args)
    



