import torch
import argparse
from sklearn.metrics import r2_score
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
def get_args():
    parser = argparse.ArgumentParser(description='Graph rationalization with Environment-based Augmentation')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # model
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--use_linear_predictor', default=False, action='store_true',
                        help='Use Linear predictor')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='size ratio to regularize the rationale subgraph (default: 0.4)')

    # training
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--l2reg', type=float, default=5e-6,
                        help='L2 norm (default: 5e-6)')
    parser.add_argument('--use_lr_scheduler', default=False, action='store_true',
                        help='Use learning rate scheduler CosineAnnealingLR')
    parser.add_argument('--use_clip_norm', default=False, action='store_true',
                        help='Use learning rate clip norm')
    parser.add_argument('--path_list', nargs="+", default=[1,4],
                        help='path for alternative optimization')
    parser.add_argument('--initw_name', type=str, default='default',
                        choices=['default','orthogonal','normal','xavier','kaiming'],
                        help='method name to initialize neural weights')

    parser.add_argument('--dataset', type=str, default="ogbg-molbbbp",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--trails', type=int, default=5,
                        help='numer of experiments (default: 5)')
    parser.add_argument('--by_default', default=False, action='store_true',
                        help='use default configuration for hyperparameters')
    ## 自己补充的重要参数

    parser.add_argument('--beta_infonce', type=float, default=0.001)
    parser.add_argument('--beta_club', type=float, default=0.001)
    parser.add_argument('--date', type=str, default='0402')
    parser.add_argument('--model_name', type=str, default='Graph_Student', help='model name')
    parser.add_argument('--train_type', type=str, default='student', help='model name')

    parser.add_argument('--algin_loss', type=float, default=1)
    parser.add_argument('--client_number', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=4.0)
    
    ## only for scis

    args = parser.parse_args()
    
    return args

cls_criterion = torch.nn.BCEWithLogitsLoss()
CELoss = torch.nn.CrossEntropyLoss(reduction="mean")
reg_criterion = torch.nn.MSELoss()

def get_kmeans(args, model, device, loader):
    model.eval()
    all_env = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                h_env = model.get_kmeans_forward(batch)
                all_env.append(h_env)
    all_env_tensor = torch.cat(all_env,0)
    num_clusters = 5
    # kmeans
    if num_clusters == 1:
        cluster_centers = all_env_tensor.mean(dim=0, keepdim=True)
        cluster_ids_x = torch.zeros(all_env_tensor.size(0), dtype=torch.long)
    else:
        cluster_ids_x, cluster_centers = kmeans(X=all_env_tensor, num_clusters=num_clusters, distance='euclidean',device = device)

    return cluster_ids_x, cluster_centers,num_clusters

def train_xmrl(args, model, device, loader, optimizers, task_type, optimizer_name,loss_logger,communication_env=None,server_model=None):
    optimizer = optimizers[optimizer_name]
    model.train()
    if optimizer_name == 'predictor':
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=True)
        set_requires_grad([model.separator], requires_grad=False)
    if optimizer_name == 'separator':
        set_requires_grad([model.separator], requires_grad=True)
        set_requires_grad([model.graph_encoder,model.predictor], requires_grad=False)
        
    for step, batch in enumerate(loader):
        # if step>10:
        #     break
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            if communication_env == None:
                pred = model(batch)
            else:
                server_model.eval()
                pos = server_model.get_pos(batch).detach()
                pred = model(batch,pos,communication_env)
            if "classification" in task_type:
                criterion = cls_criterion
            else:
                criterion = reg_criterion

            target = batch.y.to(torch.float32)
            is_labeled = batch.y == batch.y
            loss = criterion(pred['pred_rem'].to(torch.float32)[is_labeled], target[is_labeled]) 
            if communication_env != None:
                target_rep = batch.y.to(torch.float32).repeat_interleave(len(communication_env),dim=0)
                is_labeled_rep = target_rep == target_rep
                loss += criterion(pred['pred_rep'].to(torch.float32)[is_labeled_rep], target_rep[is_labeled_rep])
                loss += args.algin_loss * pred['loss_contrastive']
            

            # loss += pred['loss_reg']
            if optimizer_name == 'separator': 
                loss += pred['loss_reg']

            loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def eval_test(args, model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.eval_forward(batch)
    
            if args.dataset.startswith('plym'):
                if args.plym_prop == 'density' :
                    batch.y = torch.log(batch[args.plym_prop])
                else:
                    batch.y = batch[args.plym_prop]
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.dataset.startswith('plym'):
        return [evaluator.eval(input_dict)['rmse'], r2_score(y_true, y_pred)]
    elif args.dataset.startswith('ogbg'):
        return [evaluator.eval(input_dict)['rocauc']]

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


