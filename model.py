import os.path

import torch
torch.set_num_threads(8)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloader import DataSet
from lightGCN import LightGCN
import scipy.sparse as sp


class UniMBR(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(UniMBR, self).__init__()
        self.dataset = dataset
        self.device = args.device
        self.layers = args.layers
        self.con = args.con
        self.gen = args.gen
        self.l2 = args.decay
        self.c_temp = args.temp
        self.lambda_s = args.lambda_s
        self.neg_edge = args.neg_edge
        self.dropout = args.dropout
        
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix # interaction matrix [b1_intermatrix, b2_intermaterix,...,]
        self.all_inter_matrix = dataset.all_inter_matrix # global interaction matrix

        self.test_users = list(dataset.test_interacts.keys()) 
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size 
        self.user_embedding_glo = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding_glo = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        
        self.user_embedding_loc = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)       
        self.item_embedding_loc = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        if len(dataset.inter_matrix) == 3:
            self.aux_graph_view = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[0])
            self.aux_graph_cart = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[1])
                        
        else:
            self.aux_graph_view = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[0])
            self.aux_graph_coll = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[1])
            self.aux_graph_cart = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[2])
                        
        self.tar_graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[-1])
        self.glo_graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.all_inter_matrix)
        
        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        
        self._load_model()


    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def bpr_loss(self, p_score, n_score):
        self.gamma = 1e-10
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss

    def con_loss(self, pos, aug):
        #pos = pos[:, 0, :]
        #aug = aug[:, 0, :]
        sampled_indicies = torch.randperm(pos.shape[0])[:1024] # batch_size sampling
        
        pos = pos[sampled_indicies, :]
        aug = aug[sampled_indicies, :]
        
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos*aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1,0))
        
        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis = 1)
        
        c_loss = -torch.mean(torch.log(pos_score / ttl_score))
        
        return c_loss
        
    
    def gen_loss(self, user, item, adj, batch_size=1024):
        num_neg_samples = self.neg_edge
        adj = adj.tocoo()
    
        coo_indices = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
        coo_values = torch.tensor(adj.data, dtype=torch.float32)
    
        coo_indices = coo_indices.to(self.device)
        coo_values = coo_values.to(self.device)
    
        num_pos_samples = coo_indices.size(1)
    
        if batch_size < num_pos_samples:
            sampled_indices = torch.randint(0, num_pos_samples, (batch_size,), device=self.device)
        else:
            sampled_indices = torch.arange(0, num_pos_samples, device=self.device)
    
        pos_user_indices = coo_indices[0][sampled_indices]
        pos_item_indices = coo_indices[1][sampled_indices]

        pos_scores = torch.sigmoid((user[pos_user_indices] * item[pos_item_indices]).sum(dim=1))

        neg_user_indices = torch.randint(0, user.size(0), (batch_size * num_neg_samples,), device=coo_indices.device)
        neg_item_indices = torch.randint(0, item.size(0), (batch_size * num_neg_samples,), device=coo_indices.device)

        neg_scores = torch.sigmoid((user[neg_user_indices] * item[neg_item_indices]).sum(dim=1))

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])

        bce_loss = F.binary_cross_entropy(all_scores, all_labels)
    
        return bce_loss
    
    
    def forward(self, batch_data):
        if len(self.dataset.inter_matrix) == 3:
            
            embeddings_loc = torch.cat([self.user_embedding_loc.weight, self.item_embedding_loc.weight], dim=0)
            embeddings_glo = torch.cat([self.user_embedding_glo.weight, self.item_embedding_glo.weight], dim=0)
            
            view_embeddings = self.aux_graph_view(embeddings_loc)
            cart_embeddings = self.aux_graph_cart(embeddings_loc)
            tar_embeddings = self.tar_graph(embeddings_loc) 
            
            glo_embeddings = self.glo_graph(embeddings_glo)
            glo_embeddings_aug = self.glo_graph_aug(embeddings_glo)
            
            user_view_embedding, item_view_embedding = torch.split(view_embeddings, [self.n_users + 1, self.n_items + 1])
            user_cart_embedding, item_cart_embedding = torch.split(cart_embeddings, [self.n_users + 1, self.n_items + 1])
            user_tar_embedding, item_tar_embedding = torch.split(tar_embeddings, [self.n_users + 1, self.n_items + 1])
            
            user_glo_embedding, item_glo_embedding = torch.split(glo_embeddings, [self.n_users + 1, self.n_items + 1])
            user_glo_embedding_aug, item_glo_embedding_aug = torch.split(glo_embeddings_aug, [self.n_users + 1, self.n_items + 1]) # augmented embeddings
            
            c_loss_user = self.con_loss(user_glo_embedding, user_glo_embedding_aug)
            c_loss_item = self.con_loss(item_glo_embedding, item_glo_embedding_aug)
            
            c_loss = (c_loss_user + c_loss_item) / 2
            
            bce_loss_rv = (self.gen_loss(user_view_embedding, item_view_embedding, self.inter_matrix[-1]) + self.gen_loss(user_cart_embedding, item_cart_embedding, self.inter_matrix[0]) + self.gen_loss(user_tar_embedding, item_tar_embedding, self.inter_matrix[1])) / 3
            
            bce_loss_fw = (self.gen_loss(user_view_embedding, item_view_embedding, self.inter_matrix[1]) + self.gen_loss(user_cart_embedding, item_cart_embedding, self.inter_matrix[-1]) + self.gen_loss(user_tar_embedding, item_tar_embedding, self.inter_matrix[0])) / 3
            
            bce_loss = (bce_loss_rv + bce_loss_fw) / 2
            
            user_loc_embedding = (user_view_embedding +  user_cart_embedding + user_tar_embedding) / 3
            item_loc_embedding = (item_view_embedding +  item_cart_embedding + item_tar_embedding) / 3
            
            pair_samples = batch_data[:, -1, :-1] 
            
            mask = torch.any(pair_samples != 0, dim=-1) 
            pair_samples = pair_samples[mask] 
            
            bpr_loss = 0
            
            if pair_samples.shape[0] > 0:
                user_samples = pair_samples[:, 0].long()
                item_samples = pair_samples[:, 1:].long()
                u_emb_glo = user_glo_embedding[user_samples].unsqueeze(1)
                i_emb_glo = item_glo_embedding[item_samples]
                score_point_glo = torch.sum((u_emb_glo * i_emb_glo), dim=-1)
                
                u_emb_loc = user_loc_embedding[user_samples].unsqueeze(1)
                i_emb_loc = item_loc_embedding[item_samples]
                score_point_loc = torch.sum((u_emb_loc * i_emb_loc), dim = -1)
                
                bpr_scores = self.lambda_s*score_point_glo + (1-self.lambda_s)*score_point_loc
                p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
                bpr_loss += self.bpr_loss(p_scores, n_scores)
            
            loss = bpr_loss + self.gen * bce_loss + self.con * c_loss
        else:
            embeddings_loc = torch.cat([self.user_embedding_loc.weight, self.item_embedding_loc.weight], dim=0)
            embeddings_glo = torch.cat([self.user_embedding_glo.weight, self.item_embedding_glo.weight], dim=0)
            
            view_embeddings = self.aux_graph_view(embeddings_loc)
            coll_embeddings = self.aux_graph_coll(embeddings_loc)
            cart_embeddings = self.aux_graph_cart(embeddings_loc)
            tar_embeddings = self.tar_graph(embeddings_loc) 
            
            glo_embeddings = self.glo_graph(embeddings_glo)
            glo_embeddings_aug = self.glo_graph_aug(embeddings_glo)
            
            user_view_embedding, item_view_embedding = torch.split(view_embeddings, [self.n_users + 1, self.n_items + 1])
            user_coll_embedding, item_coll_embedding = torch.split(coll_embeddings, [self.n_users + 1, self.n_items + 1])
            user_cart_embedding, item_cart_embedding = torch.split(cart_embeddings, [self.n_users + 1, self.n_items + 1])
            user_tar_embedding, item_tar_embedding = torch.split(tar_embeddings, [self.n_users + 1, self.n_items + 1])
            
            user_glo_embedding, item_glo_embedding = torch.split(glo_embeddings, [self.n_users + 1, self.n_items + 1])
            user_glo_embedding_aug, item_glo_embedding_aug = torch.split(glo_embeddings_aug, [self.n_users + 1, self.n_items + 1])
            
            c_loss_user = self.con_loss(user_glo_embedding, user_glo_embedding_aug)
            c_loss_item = self.con_loss(item_glo_embedding, item_glo_embedding_aug)
            
            c_loss = (c_loss_user + c_loss_item) / 2
            
            bce_loss_rv = (self.gen_loss(user_view_embedding, item_view_embedding, self.inter_matrix[-1]) + self.gen_loss(user_coll_embedding, item_coll_embedding, self.inter_matrix[0]) + self.gen_loss(user_cart_embedding, item_cart_embedding, self.inter_matrix[1]) + self.gen_loss(user_tar_embedding, item_tar_embedding, self.inter_matrix[2])  + self.gen_loss(user_tar_embedding, item_tar_embedding, self.inter_matrix[1]) + self.gen_loss(user_cart_embedding, item_cart_embedding, self.inter_matrix[0])) / 6
            
            bce_loss_fw = (self.gen_loss(user_view_embedding, item_view_embedding, self.inter_matrix[1]) + self.gen_loss(user_coll_embedding, item_coll_embedding, self.inter_matrix[2]) + self.gen_loss(user_cart_embedding, item_cart_embedding, self.inter_matrix[-1]) + self.gen_loss(user_tar_embedding, item_tar_embedding, self.inter_matrix[0]) + self.gen_loss(user_view_embedding, item_view_embedding, self.inter_matrix[2]) + self.gen_loss(user_coll_embedding, item_coll_embedding, self.inter_matrix[-1])) / 6
            
            bce_loss =  (bce_loss_rv + bce_loss_fw) / 2
            
            pair_samples = batch_data[:, -1, :-1] 
            
            user_loc_embedding = (user_view_embedding + user_coll_embedding + user_cart_embedding + user_tar_embedding) / 4
            item_loc_embedding = (item_view_embedding + item_coll_embedding + item_cart_embedding + item_tar_embedding) / 4
            
            mask = torch.any(pair_samples != 0, dim=-1) 
            pair_samples = pair_samples[mask] 
            
            bpr_loss = 0
            
            if pair_samples.shape[0] > 0:
                user_samples = pair_samples[:, 0].long()
                item_samples = pair_samples[:, 1:].long()
                u_emb_glo = user_glo_embedding[user_samples].unsqueeze(1)
                i_emb_glo = item_glo_embedding[item_samples]
                score_point_glo = torch.sum((u_emb_glo * i_emb_glo), dim=-1)
                
                u_emb_loc = user_loc_embedding[user_samples].unsqueeze(1)
                i_emb_loc = item_loc_embedding[item_samples]
                score_point_loc = torch.sum((u_emb_loc * i_emb_loc), dim = -1)

                bpr_scores = self.lambda_s*score_point_glo + (1-self.lambda_s)*score_point_loc
                p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
                bpr_loss += self.bpr_loss(p_scores, n_scores)
                
            loss = bpr_loss + self.gen*bce_loss + self.con * c_loss
            
        return loss

    def full_predict(self, users):
        embeddings_loc = torch.cat([self.user_embedding_loc.weight, self.item_embedding_loc.weight], dim=0)
        embeddings_glo = torch.cat([self.user_embedding_glo.weight, self.item_embedding_glo.weight], dim=0)
        
        if len(self.dataset.inter_matrix) == 3:
            view_embeddings = self.aux_graph_view(embeddings_loc)
            cart_embeddings = self.aux_graph_cart(embeddings_loc)
            tar_embeddings = self.tar_graph(embeddings_loc) 
            glo_embeddings = self.glo_graph(embeddings_glo) 
            
            user_view_embedding, item_view_embedding = torch.split(view_embeddings, [self.n_users + 1, self.n_items + 1])
            user_cart_embedding, item_cart_embedding = torch.split(cart_embeddings, [self.n_users + 1, self.n_items + 1])
            user_tar_embedding, item_tar_embedding = torch.split(tar_embeddings, [self.n_users + 1, self.n_items + 1])
            
            user_glo_embedding, item_glo_embedding = torch.split(glo_embeddings, [self.n_users + 1, self.n_items + 1])
        
            user_emb_loc = (user_view_embedding[users.long()] + user_cart_embedding[users.long()] + user_tar_embedding[users.long()]) / 3
            user_emb_glo = user_glo_embedding[users.long()]
        
            item_loc_embedding = (item_view_embedding + item_cart_embedding + item_tar_embedding) / 3
        
            scores_loc = torch.matmul(user_emb_loc, item_loc_embedding.transpose(0, 1))
            scores_glo = torch.matmul(user_emb_glo, item_glo_embedding.transpose(0, 1))
        
            scores =  self.lambda_s*scores_glo + (1-self.lambda_s)*scores_loc
        
        else:
            view_embeddings = self.aux_graph_view(embeddings_loc)
            coll_embeddings = self.aux_graph_coll(embeddings_loc)
            cart_embeddings = self.aux_graph_cart(embeddings_loc)
            tar_embeddings = self.tar_graph(embeddings_loc) # Pretrai
            
            glo_embeddings = self.glo_graph(embeddings_glo) # target behavior에 대한 graph
            
            user_view_embedding, item_view_embedding = torch.split(view_embeddings, [self.n_users + 1, self.n_items + 1])
            user_coll_embedding, item_coll_embedding = torch.split(coll_embeddings, [self.n_users + 1, self.n_items + 1])
            user_cart_embedding, item_cart_embedding = torch.split(cart_embeddings, [self.n_users + 1, self.n_items + 1])
            user_tar_embedding, item_tar_embedding = torch.split(tar_embeddings, [self.n_users + 1, self.n_items + 1])
            
            user_glo_embedding, item_glo_embedding = torch.split(glo_embeddings, [self.n_users + 1, self.n_items + 1])
        
            user_emb_loc = (user_view_embedding[users.long()] + user_coll_embedding[users.long()] + user_cart_embedding[users.long()] + user_tar_embedding[users.long()]) / 4
            user_emb_glo = user_glo_embedding[users.long()]
        
            item_loc_embedding = (item_view_embedding + item_coll_embedding + item_cart_embedding + item_tar_embedding) / 4
        
            scores_loc = torch.matmul(user_emb_loc, item_loc_embedding.transpose(0, 1))
            scores_glo = torch.matmul(user_emb_glo, item_glo_embedding.transpose(0, 1))
        
            scores = self.lambda_s*scores_glo + (1-self.lambda_s)*scores_loc
            
        return scores

