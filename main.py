import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(8)
from loguru import logger

from dataloader import DataSet
from model import MEMBER

from trainer import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=32, help='')
    parser.add_argument('--con_s', type=float, default=0.1, help='') # contrastive loss weight
    parser.add_argument('--con_us', type=float, default=0.1, help='') # contrastive loss weight
    
    parser.add_argument('--temp_s', type = float, default = 0.6) # contrastive loss : temperature hyperparameter
    parser.add_argument('--temp_us', type = float, default = 0.6) # contrastive loss : temperature hyperparameter
    
    parser.add_argument('--gen', type=float, default=0.5, help='') # generative loss weight
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--layers_sg', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2) # Random edge dropout ratio
    parser.add_argument('--lambda_s', type=float, default=0.5) # weight for balancing global and local scores for visited-item expert
    parser.add_argument('--lambda_us', type=float, default=0.5) # weight for balancing global and local scores for unvisited-item expert
    
    parser.add_argument('--data_name', type=str, default='tmall', help='') # data name 
    parser.add_argument('--behaviors', help='', action='append')

    parser.add_argument('--neg_count', type=int, default=1)
    parser.add_argument('--neg_edge', type=int, default=3) # ratio for negative edges sampling in bi-directional view generation

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg', 'recall'], help='')
    parser.add_argument('--alpha', type=int, default=1, help='')
    
    parser.add_argument('--lr', type=float, default=0.005, help='')    
    parser.add_argument('--decay', type=float, default=1e-7, help='')
    
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, default='tmall', help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--setting', type=str, default='basic', help='basic, visited, unvisited')
    
    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']

    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
        
    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['view', 'cart', 'buy']

    else:
        raise Exception('data_name cannot be None')

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model_visited = MEMBER(args, dataset, expert_type = 'visited').to(args.device)    
    model_unvisited = MEMBER(args, dataset, expert_type = 'unvisited').to(args.device)
    
    trainer = Trainer(model_visited, model_unvisited, dataset, args)
    logger.info(args.__str__())
    
    trainer.train_model()
    logger.info('train end total cost time: {}'.format(time.time() - start))
