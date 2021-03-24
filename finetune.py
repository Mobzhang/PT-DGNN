import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PT-DGNN.data import *
from PT-DGNN.model import *
from torch_geometric.utils import negative_sampling, remove_self_loops, train_test_split_edges
from torch_geometric.data import Data
from warnings import filterwarnings
filterwarnings("ignore")
import networkx as nx
import random
import gc
from collections import defaultdict
from  sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import argparse

parser = argparse.ArgumentParser(description='Fine-Tuning on link prediction task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./datadrive/dataset',
                    help='The address of preprocessed graph.')
parser.add_argument('--data_name', type=str, default='hepph',
                    help='The address of preprocessed graph.')
parser.add_argument('--time', type=int, default=1,
                    help='The network has timestamp.')
parser.add_argument('--use_pretrain', type=int, help='Whether to use pre-trained model', default=1)
parser.add_argument('--pretrain_model_dir', type=str, default='./datadrive/models',
                    help='The address for pretrained model.')
parser.add_argument('--model_dir', type=str, default='./datadrive/models',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')     
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')

'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='gcn',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')


'''
    Optimization arguments
'''
parser.add_argument('--max_lr', type=float, default=1e-3,
                    help='Maximum learning rate.')
parser.add_argument('--scheduler', type=str, default='cycle',
                    help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
parser.add_argument('--n_epoch', type=int, default=20,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient Norm Clipping') 

args = parser.parse_args()
args_print(args)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print('Start Loading Graph Data...')
graph = dill.load(open(os.path.join(args.data_dir, args.data_name + '.pk'), 'rb'))
print('Finish Loading Graph Data!')

target_type = 'def'
rel_stop_list = ['self']
train_target_nodes = graph.train_target_nodes
valid_target_nodes = graph.valid_target_nodes
test_target_nodes  = graph.test_target_nodes

train_target_nodes = np.concatenate([train_target_nodes, np.ones(len(train_target_nodes))]).reshape(2, -1).transpose()
valid_target_nodes = np.concatenate([valid_target_nodes, np.ones(len(valid_target_nodes))]).reshape(2, -1).transpose()
test_target_nodes = np.concatenate([test_target_nodes, np.ones(len(test_target_nodes))]).reshape(2, -1).transpose()

types = graph.get_types()

def link_prediction_sample(seed, target_nodes, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    
    np.random.seed(seed)
    samp_target_nodes = target_nodes[np.random.choice(len(target_nodes), args.batch_size)]
    feature, times, edge_list, _, attr = sample_subgraph(graph, time_range, \
                inp = {target_type: samp_target_nodes}, feature_extractor = feature_reddit, \
                    sampled_depth = args.sample_depth, sampled_number = args.sample_width, ist = args.time)
    if args.time:
        temp_list = np.array(edge_list['def']['def']['def'],dtype=float)
        edge_list['def']['def']['def'] = list(temp_list[:, :-1])
        edge_list['def']['def']['time'] = list(temp_list)  ### temporal GraphSAGE loss input temporal graph
#         print(edge_list['def']['def']['time'])

    # sample pairs  new!!!!!!!            
    node_feature, node_type, edge_time, edge_index, edge_type, node_positive_pairs, node_negative_pairs, node_dict, edge_dict  = \
            to_torch(feature, times, edge_list, graph, num_neg=10)

    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, node_positive_pairs, node_negative_pairs


def get_roc_score(edges_pos, edges_neg, emb=None):


    adj_rec = np.dot(emb, emb.T)
    preds = []
    for e in edges_pos:
        preds.append((adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append((adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg]).reshape(-1,1)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    
    x_train, x_valid, y_train, y_valid = train_test_split(preds_all, labels_all, test_size=.20, random_state=9)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_valid_pred_prob = lr.predict_proba(x_valid)[:,1]
    y_valid_pred_01 = lr.predict(x_valid)
    auc = roc_auc_score(y_valid, y_valid_pred_prob)
    
    return auc


def prepare_data(pool):
    jobs = []
    for _ in np.arange(args.n_batch - 1):
        jobs.append(pool.apply_async(link_prediction_sample, args=(randint(), train_target_nodes, {1: True})))
    jobs.append(pool.apply_async(link_prediction_sample, args=(randint(), valid_target_nodes, {1: True})))
    return jobs

stats = []
res = []
best_val   = 10000
train_step = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)


gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]), n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = False)


    
if args.use_pretrain:
    if  args.time:
        gnn.load_state_dict(load_gnn(torch.load(os.path.join(args.pretrain_model_dir, 'gpt_all_' + args.data_name))), strict = False)
        print('Load Pre-trained Model from (%s)' % os.path.join(args.pretrain_model_dir, 'gpt_all_' + args.data_name))
    else:
        gnn.load_state_dict(load_gnn(torch.load(os.path.join(args.pretrain_model_dir, 'gpt_all_no_t_' + args.data_name))), strict = False) 
        print('Load Pre-trained Model from (%s)' % os.path.join(args.pretrain_model_dir, 'gpt_all_no_t_' + args.data_name))
# print('Load Pre-trained Model from (%s)' % args.pretrain_model_dir)

gnn = gnn.to(device)

optimizer = torch.optim.AdamW(gnn.parameters(), lr = 5e-4)



if args.scheduler == 'cycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.02, anneal_strategy='linear', final_div_factor=100,\
                        max_lr = args.max_lr, total_steps = args.n_batch * args.n_epoch + 1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)


for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train
    '''
    gnn.train()
    train_losses = []
    for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, node_positive_pairs, node_negative_pairs in train_data:

        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))


        loss = gnn.get_loss_sage(node_rep, node_positive_pairs, node_negative_pairs, device)
#         print('loss device: ',  loss.device)
#         print('train loss', loss)
        optimizer.zero_grad() 
        torch.cuda.empty_cache()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(gnn.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        del  loss
    '''
        Valid
    '''
#     print('len(loss): ', len(train_losses))
    gnn.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, node_positive_pairs, node_negative_pairs = valid_data

        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))

        loss = gnn.get_loss_sage(node_rep, node_positive_pairs, node_negative_pairs, device)
        
      
        '''
            Calculate Valid F1. Update the best model based on highest F1 score.
        '''

        if loss < best_val:
            best_val = loss
            if args.time:
                torch.save(gnn.state_dict(), os.path.join(args.model_dir, args.data_name + '_' + args.conv_name))
            else:
                torch.save(gnn.state_dict(), os.path.join(args.model_dir, 'no_t_' + args.data_name + '_' + args.conv_name))
            print('UPDATE!!!')
        
        st = time.time()

        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  ") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                    loss.cpu().detach().tolist()))
        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del  loss
    del train_data, valid_data    
    gc.collect()
    


#  load test model
if args.time:
#     best_model = gnn.load_state_dict(torch.load(os.path.join(args.model_dir, args.data_name + '_' + args.conv_name)))
    gnn.load_state_dict(torch.load(os.path.join(args.model_dir, args.data_name + '_' + args.conv_name)))
    print('Load best test Model from (%s)' %  os.path.join(args.model_dir, args.data_name + '_' + args.conv_name))
else:
#     best_model = gnn.load_state_dict(torch.load(os.path.join(args.model_dir, 'no_t_' + args.data_name + '_' + args.conv_name)))
    gnn.load_state_dict(torch.load(os.path.join(args.model_dir, 'no_t_' + args.data_name + '_' + args.conv_name)))
    print('Load best test Model from (%s)' %  os.path.join(args.model_dir, 'no_t_' + args.data_name + '_' + args.conv_name))
# best_model.eval()


with torch.no_grad():
    test_res = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, _, _= \
                    link_prediction_sample(randint(), test_target_nodes, {1: True})
        node_emb = gnn.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), \
                                   edge_index.to(device), edge_type.to(device))
        
        G = nx.DiGraph()
        G.add_edges_from(edge_index.t().tolist())
        print(G.number_of_edges())
        n_edges = int(G.number_of_edges()*0.2)
        print('number of test edges :', n_edges)
        edges_pos = random.sample(list(G.edges()), n_edges)
        edges_neg = random.sample(list(nx.non_edges(G)), n_edges) 
        
        test_roc = get_roc_score(edges_pos, edges_neg, node_emb.cpu().numpy())
        test_res += [test_roc]
#         print('done!')
    print('Best Test auc: %.4f' % np.average(test_res))
