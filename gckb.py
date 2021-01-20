# -*- coding: utf-8 -*-
"""GCKB

The codes of data loading, preprocessing and evaluation refer to https://github.com/deepakn97/relationPrediction
benchmark data: FB15k-237
"""
# Commented out IPython magic to ensure Python compatibility.
# %%
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

import random
import argparse
import os
import sys
import logging
import time
import pickle

# %%
args = argparse.ArgumentParser()
# network arguments
args.add_argument("-data", "--data",
                  default="./data/FB15k-237/", help="data directory")
args.add_argument("-e_c", "--epochs", type=int,
                  default=200, help="Number of epochs")
args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                  default=1e-6, help="L2 reglarization for conv")
args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                  default=True, help="Use pretrained embeddings")
args.add_argument("-emb_size", "--embedding_size", type=int,
                  default=100, help="Size of embeddings (if pretrained not used)")
args.add_argument("-l", "--lr", type=float, default=5e-4)
args.add_argument("-outfolder", "--output_folder",
                  default="./checkpoints/fb/out/", help="Folder name to save the models.")

# arguments for convolution network
args.add_argument("-b_s", "--batch_size", type=int,
                  default=256, help="Batch size for conv")
args.add_argument("-alpha", "--alpha", type=float,
                  default=0.0, help="LeakyRelu alphas for conv layer")
args.add_argument("-neg_s_conv", "--valid_invalid_ratio", type=int, default=60,
                  help="Ratio of valid to invalid triples for convolution training")
args.add_argument("-o", "--out_channels", type=int, default=50,
                  help="Number of output channels in conv layer")
args.add_argument("-drop", "--drop", type=float,
                  default=0.3, help="Dropout probability for convolution layer")

args = args.parse_args()
args.pretrained_emb = True
print(args)

# %%


def read_entity_from_id(filename='./data/FB15k-237/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id

# %%


def read_relation_from_id(filename='./data/FB15k-237/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

# %%


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

# %%


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

# %%
def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []   # [h, r, t]
    
    if args.add_inverse:
        num_rel = len(relation2id)
    
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(entity2id[e1])
        unique_entities.add(entity2id[e2])
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        
        if args.add_inverse:
            triples_data.append(
                (entity2id[e2], relation2id[relation]+num_rel, entity2id[e1]))

    print("number of unique_entities ->", len(unique_entities))
    # the triples, adjacency matrix, number entities.
    return triples_data, list(unique_entities)

# %%
def build_data(path='./data/FB15k-237/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)
    
    
    lef = {}    # (h, r) --> [t,]
    rig = {}    # (r, t) --> [h,]
    rel_lef = {}    # r -->{h,}
    rel_rig = {}    # r -->{t,}
    
    
    all_triples = train_triples + validation_triples + test_triples
    for triple in all_triples:
        h, r, t = triple
        if not (h, r) in lef:
            lef[(h, r)] = []
        if not (r, t) in rig:
            rig[(r, t)] = []
        lef[(h, r)] += [t]
        rig[(r, t)] += [h]
        if not r in rel_lef:
            rel_lef[r] = set()
        if not r in rel_rig:
            rel_rig[r] = set()
        rel_lef[r].add(h)
        rel_rig[r].add(t)
    
    with open(os.path.join(path, 'constraint.txt'), 'w') as f:
        for i in rel_lef:
            f.write(f'{i}\t{len(rel_lef[i])}')
            for j in rel_lef[i]:
                f.write(f'\t{j}')
            f.write('\n')
            
            f.write(f'{i}\t{len(rel_rig[i])}')
            for j in rel_rig[i]:
                f.write(f'\t{j}')
            f.write('\n')
    
    
    rel_lef = {}     # r --> the out degree
    tot_lef = {}     # r --> total the number of (X, r) in the lef key.
    rel_rig = {}     # r --> the in degree
    tot_rig = {}     # r --> total the number of (r, X) in the rig key.
    # lef: {(h, r): t}
    # rig: {(r, t): h}
    for i in lef:
        if not i[1] in rel_lef:
            rel_lef[i[1]] = 0
            tot_lef[i[1]] = 0
        rel_lef[i[1]] += len(lef[i])
        tot_lef[i[1]] += 1.0

    for i in rig:
        if not i[0] in rel_rig:
            rel_rig[i[0]] = 0
            tot_rig[i[0]] = 0
        rel_rig[i[0]] += len(rig[i])
        tot_rig[i[0]] += 1.0

    f11 = open(os.path.join(path, "1-1.txt"), "w")
    f1n = open(os.path.join(path, "1-n.txt"), "w")
    fn1 = open(os.path.join(path, "n-1.txt"), "w")
    fnn = open(os.path.join(path, "n-n.txt"), "w")
    fall = open(os.path.join(path, "test2id_all.txt"), "w")
    for triple in test_triples:
        h, r, t = triple
        content = f'{h}\t{r}\t{t}\n'
        rig_n = rel_lef[r] / tot_lef[r]
        lef_n = rel_rig[r] / tot_rig[r]
        if (rig_n < 1.5 and lef_n < 1.5):
            f11.write(content)
            fall.write("0"+"\t"+content)
        if (rig_n >= 1.5 and lef_n < 1.5):
            f1n.write(content)
            fall.write("1"+"\t"+content)
        if (rig_n < 1.5 and lef_n >= 1.5):
            fn1.write(content)
            fall.write("2"+"\t"+content)
        if (rig_n >= 1.5 and lef_n >= 1.5):
            fnn.write(content)
            fall.write("3"+"\t"+content)
    fall.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return train_triples, validation_triples, test_triples, \
        entity2id, relation2id, headTailSelector, unique_entities_train


"""
读取数据，加载embedding

train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train

entity_embeddings, relation_embeddings"""
# %%
train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
    args.data, is_unweigted=False, directed=True)
num_entities = len(entity2id)
num_relations = len(relation2id)

# if args.pretrained_emb and not args.add_inverse:
#     print(args.pretrained_emb)
print(args.pretrained_emb and not args.add_inverse)
entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                            os.path.join(args.data, 'relation2vec.txt'))
print("Initialised relations and entities from TransE")

# else:
    # entity_embeddings = np.random.randn(len(entity2id), args.embedding_size)
    # relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)
    # if args.add_inverse:
    #     relation_embeddings = np.random.randn(2*len(relation2id), args.embedding_size)
    # print("Initialised relations and entities randomly")
# entity_embeddings = np.random.randn(len(entity2id), args.embedding_size)
# relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)
# if args.add_inverse:
#     relation_embeddings = np.random.randn(2*len(relation2id), args.embedding_size)
# print("Initialised relations and entities randomly")
entity_embeddings = torch.from_numpy(entity_embeddings)
relation_embeddings = torch.from_numpy(relation_embeddings)

train_triples = train_data
train_indices = np.array(list(train_triples)).astype(np.int32)
train_values = np.array([[1, 1, 1]] * len(train_triples)).astype(np.float32)

validation_triples = validation_data
validation_indices = np.array(list(validation_triples)).astype(np.int32)
validation_values = np.array(
    [[1]] * len(validation_triples)).astype(np.float32)

test_triples = test_data
test_indices = np.array(list(test_triples)).astype(np.int32)
test_values = np.array([[1]] * len(test_triples)).astype(np.float32)

valid_triples_dict = {j: i for i, j in enumerate(
    train_triples + validation_triples + test_triples)}

# %%
def get_iteration_batch(iter_num, batch_size, invalid_valid_ratio, num_entities):
    if (iter_num + 1) * batch_size <= len(train_indices):
        batch_indices = np.empty(
            (batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
        batch_values = np.empty(
            (batch_size * (invalid_valid_ratio + 1), 3)).astype(np.float32)

        indices = range(batch_size * iter_num, batch_size * (iter_num + 1))

        # 获得batch训练数据
        batch_indices[:batch_size, :] = train_indices[indices, :]
        batch_values[:batch_size, :] = train_values[indices, :]

        last_index = batch_size

        if invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, num_entities, last_index * invalid_valid_ratio)

            batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                batch_indices[:last_index, :], (invalid_valid_ratio, 1))
            batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                batch_values[:last_index, :], (invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(invalid_valid_ratio // 2):
                    current_index = i * (invalid_valid_ratio // 2) + j

                    while (random_entities[current_index], batch_indices[last_index + current_index, 1],
                           batch_indices[last_index + current_index, 2]) in valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, num_entities)
                    batch_indices[last_index + current_index,
                                  0] = random_entities[current_index]
                    batch_values[last_index + current_index, :] = [-1, -1, 1]

                for j in range(invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (invalid_valid_ratio // 2) + \
                        (i * (invalid_valid_ratio // 2) + j)

                    while (batch_indices[last_index + current_index, 0], batch_indices[last_index + current_index, 1],
                           random_entities[current_index]) in valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, num_entities)
                    batch_indices[last_index + current_index,
                                  2] = random_entities[current_index]
                    batch_values[last_index + current_index, :] = [-1, 1, -1]

            return batch_indices, batch_values

        return batch_indices, batch_values

    else:
        last_iter_size = len(train_indices) - \
            batch_size * iter_num
        batch_indices = np.empty(
            (last_iter_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
        batch_values = np.empty(
            (last_iter_size * (invalid_valid_ratio + 1), 3)).astype(np.float32)

        indices = range(batch_size * iter_num,
                        len(train_indices))
        batch_indices[:last_iter_size,
                      :] = train_indices[indices, :]
        batch_values[:last_iter_size,
                     :] = train_values[indices, :]

        last_index = last_iter_size

        if invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, num_entities, last_index * invalid_valid_ratio)

            batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                batch_indices[:last_index, :], (invalid_valid_ratio, 1))
            batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                batch_values[:last_index, :], (invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(invalid_valid_ratio // 2):
                    current_index = i * (invalid_valid_ratio // 2) + j

                    while (random_entities[current_index], batch_indices[last_index + current_index, 1],
                           batch_indices[last_index + current_index, 2]) in valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, num_entities)
                    batch_indices[last_index + current_index,
                                  0] = random_entities[current_index]
                    batch_values[last_index + current_index, :] = [-1, -1, 1]

                for j in range(invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (invalid_valid_ratio // 2) + \
                        (i * (invalid_valid_ratio // 2) + j)

                    while (batch_indices[last_index + current_index, 0], batch_indices[last_index + current_index, 1],
                           random_entities[current_index]) in valid_triples_dict.keys():
                        random_entities[current_index] = np.random.randint(
                            0, num_entities)
                    batch_indices[last_index + current_index,
                                  2] = random_entities[current_index]
                    batch_values[last_index + current_index, :] = [-1, 1, -1]

            return batch_indices, batch_values

        return batch_indices, batch_values

# %%
class GCKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha):
        super().__init__()
        self.conv_seq = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.conv_hr_a = nn.Conv2d(in_channels, out_channels, (1, input_seq_len-1))
        self.conv_rt_a = nn.Conv2d(in_channels, out_channels, (1, input_seq_len-1))

        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU(alpha)
        self.fc_layer = nn.Linear(input_dim * out_channels, 1)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2).unsqueeze(1)    # [batch_size, 1, dim, length]

        conv_hr = conv_input[:, :, :, :-1]
        conv_rt = conv_input[:, :, :, 1:]

        conv_out = self.conv_seq(conv_input)
        conv_hr = self.conv_hr_a(conv_hr)
        conv_rt = self.conv_rt_a(conv_rt)

        conv_a = torch.sigmoid(conv_hr * conv_rt)
        conv_out = self.dropout(self.non_linearity(conv_out) * conv_a)
        input_fc = conv_out.squeeze(-1).view(batch_size, -1)
        pred = self.fc_layer(input_fc)
        return pred


# %%
class Conv(nn.Module):
    def __init__(self, num_nodes, num_relations, emb_dim, dropout, alpha, out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        '''
        super().__init__()
        if args.add_inverse:
            num_relations = 2*num_relations
        self.entity_embeddings = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.relation_embeddings = nn.Parameter(torch.randn(num_relations, emb_dim))

        self.conv = GCKB(emb_dim, 3, 1, out_channels, dropout, alpha)
    
    def from_predtrain(self):
        self.entity_embeddings.data.copy_(entity_embeddings)
        self.relation_embeddings.data.copy_(relation_embeddings)

    def forward(self, batch_inputs):
        conv_input = torch.cat((self.entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.conv(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.conv(conv_input)
        return out_conv

# %%
def save_model(model, epoch, folder_name):
    print("Saving Model")
    path = os.path.join(folder_name, "trained_{}.pth".format(epoch))
    torch.save(model.state_dict(), path)
    print("Done saving Model")


# %%
CUDA = torch.cuda.is_available()

# %%


def train_conv(args):

    # Creating convolution model here.
    ####################################
    print("Only Conv model trained")
    model_conv = Conv(num_entities, num_relations, args.embedding_size,
                      args.drop, args.alpha, args.out_channels)
    
    print(entity_embeddings.size())
    print(relation_embeddings.size)

    if CUDA:
        model_conv.cuda()

    model_conv.entity_embeddings.data.copy_(entity_embeddings.cuda())
    model_conv.relation_embeddings.data.copy_(relation_embeddings.cuda())
    
    # model_conv.load_state_dict(torch.load(
        # '{0}trained_299.pth'.format(args.output_folder)), strict=False)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs))

    for epoch in range(args.epochs):
        print("\nepoch-> ", epoch)
        random.shuffle(train_triples)
        # train_indices = np.array(list(train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(train_indices) % args.batch_size == 0:
            num_iters_per_epoch = len(train_indices) // args.batch_size
        else:
            num_iters_per_epoch = (len(train_indices) // args.batch_size) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            batch_train_indices, batch_train_values = get_iteration_batch(iters, args.batch_size,
                                                                          int(args.valid_invalid_ratio), num_entities)

            if CUDA:
                batch_train_indices = Variable(
                    torch.LongTensor(batch_train_indices)).cuda()
                batch_train_values = Variable(
                    torch.FloatTensor(batch_train_values)).cuda()

            else:
                batch_train_indices = Variable(
                    torch.LongTensor(batch_train_indices))
                batch_train_values = Variable(
                    torch.FloatTensor(batch_train_values))

            pred_1 = model_conv(batch_train_indices)

            optimizer.zero_grad()

            loss = margin_loss(
                pred_1.view(-1), batch_train_values[:, 0].view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()), end='\r')

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_conv, epoch, args.output_folder)


# %%
def get_validation_pred(args, model, test_data, unique_entities, constraint=None):
    average_hits_at_100_head, average_hits_at_100_tail = [], []
    average_hits_at_ten_head, average_hits_at_ten_tail = [], []
    average_hits_at_three_head, average_hits_at_three_tail = [], []
    average_hits_at_one_head, average_hits_at_one_tail = [], []
    average_mean_rank_head, average_mean_rank_tail = [], []
    average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

    for iters in range(1):
        start_time = time.time()

        indices = [i for i in range(len(test_data))]
        batch_indices = test_data[indices, :]
        print("Sampled indices")
        print("test set length ", len(test_data))
        print()
        entity_list = [j for i, j in entity2id.items()]

        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_ten_head, hits_at_ten_tail = 0, 0
        hits_at_three_head, hits_at_three_tail = 0, 0
        hits_at_one_head, hits_at_one_tail = 0, 0

        for i in range(batch_indices.shape[0]):
            # print(i)
            start_time_it = time.time()
            new_x_batch_head = np.tile(
                batch_indices[i, :], (len(entity2id), 1))
            new_x_batch_tail = np.tile(
                batch_indices[i, :], (len(entity2id), 1))

            if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                # print(i, batch_indices[i, 0], batch_indices[i, 2])
                # print(unique_entities)
                continue

            new_x_batch_head[:, 0] = entity_list
            new_x_batch_tail[:, 2] = entity_list

            last_index_head = []  # array of already existing triples
            last_index_tail = []
            for tmp_index in range(len(new_x_batch_head)):
                temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                    new_x_batch_head[tmp_index][2])
                if temp_triple_head in valid_triples_dict.keys():
                    last_index_head.append(tmp_index)

                temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                    new_x_batch_tail[tmp_index][2])
                if temp_triple_tail in valid_triples_dict.keys():
                    last_index_tail.append(tmp_index)
                
                if constraint is not None:
                    if temp_triple_head[0] not in constraint['rel_src'][temp_triple_head[1]]:
                        last_index_head.append(tmp_index)
                    if temp_triple_tail[2] not in constraint['rel_dst'][temp_triple_tail[1]]:
                        last_index_tail.append(tmp_index)

            # Deleting already existing triples, leftover triples are invalid, according
            # to train, validation and test data
            # Note, all of them maynot be actually invalid
            new_x_batch_head = np.delete(
                new_x_batch_head, last_index_head, axis=0)
            new_x_batch_tail = np.delete(
                new_x_batch_tail, last_index_tail, axis=0)

            # adding the current valid triples to the top, i.e, index 0
            new_x_batch_head = np.insert(
                new_x_batch_head, 0, batch_indices[i], axis=0)
            new_x_batch_tail = np.insert(
                new_x_batch_tail, 0, batch_indices[i], axis=0)
            
            if args.add_inverse:
                head, rel, tail = new_x_batch_head.transpose(1, 0)
                new_x_batch_head_inverse = np.stack([tail, rel + num_relations, head])
                
                head, rel, tail = new_x_batch_tail.transpose(1, 0)
                new_x_batch_tail_inverse = np.stack([tail, rel + num_relations, head])

            import math
            # Have to do this, because it doesn't fit in memory

            if 'WN' in args.data:
                num_triples_each_shot = int(
                    math.ceil(new_x_batch_head.shape[0] / 4))

                scores1_head = model.batch_test(torch.LongTensor(
                    new_x_batch_head[:num_triples_each_shot, :]).cuda())
                scores2_head = model.batch_test(torch.LongTensor(
                    new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                scores3_head = model.batch_test(torch.LongTensor(
                    new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                scores4_head = model.batch_test(torch.LongTensor(
                    new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                scores_head = torch.cat(
                    [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
                
                if args.add_inverse:
                    num_triples_each_shot = int(
                    math.ceil(new_x_batch_head_inverse.shape[0] / 4))

                    scores1_head_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_head_inverse[:num_triples_each_shot, :]).cuda())
                    scores2_head_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_head_inverse[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_head_inverse[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_head_inverse[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    scores_head_inverse = torch.cat(
                        [scores1_head_inverse, scores2_head_inverse, scores3_head_inverse, scores4_head_inverse], dim=0)
            else:
                scores_head = model.batch_test(new_x_batch_head)
                
                if args.add_inverse:
                    scores_head_inverse = model.batch_test(new_x_batch_head_inverse)
            
            if args.add_inverse:
                scores_head = np.concatenate([scores_head, scores_head_inverse], 1)
                scores_head = np.max(scores_head, axis=-1)

            sorted_scores_head, sorted_indices_head = torch.sort(
                scores_head.view(-1), dim=-1, descending=True)
            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            ranks_head.append(
                np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
            reciprocal_ranks_head.append(1.0 / ranks_head[-1])

            # Tail part here

            if 'WN' in args.data:
                num_triples_each_shot = int(
                    math.ceil(new_x_batch_tail.shape[0] / 4))

                scores1_tail = model.batch_test(torch.LongTensor(
                    new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                scores2_tail = model.batch_test(torch.LongTensor(
                    new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                scores3_tail = model.batch_test(torch.LongTensor(
                    new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                scores4_tail = model.batch_test(torch.LongTensor(
                    new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                scores_tail = torch.cat(
                    [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)
                if args.add_inverse:
                    num_triples_each_shot = int(
                    math.ceil(new_x_batch_tail_inverse.shape[0] / 4))

                    scores1_tail_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_tail_inverse[:num_triples_each_shot, :]).cuda())
                    scores2_tail_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_tail_inverse[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_tail_inverse[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail_inverse = model.batch_test(torch.LongTensor(
                        new_x_batch_tail_inverse[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    scores_tail_inverse = torch.cat(
                        [scores1_tail_inverse, scores2_tail_inverse, scores3_tail_inverse, scores4_tail_inverse], dim=0)
            else:
                scores_tail = model.batch_test(new_x_batch_tail)
                if args.add_inverse:
                    scores_tail_inverse = model.batch_test(new_x_batch_tail_inverse)
            
            if args.add_inverse:
                scores_tail = np.concatenate([scores_tail, scores_tail_inverse], 1)
                scores_tail = np.max(scores_tail, axis=-1)

            sorted_scores_tail, sorted_indices_tail = torch.sort(
                scores_tail.view(-1), dim=-1, descending=True)

            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            ranks_tail.append(
                np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
            reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
            if i % 10 == 0:
                print(len(ranks_head), "sample - ",
                    ranks_head[-1], '\t', ranks_tail[-1], end='\r')
            # print(len(ranks_head), "sample - ", ranks_head[-1], ranks_tail[-1])

        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head = hits_at_100_head + 1
            if ranks_head[i] <= 10:
                hits_at_ten_head = hits_at_ten_head + 1
            if ranks_head[i] <= 3:
                hits_at_three_head = hits_at_three_head + 1
            if ranks_head[i] == 1:
                hits_at_one_head = hits_at_one_head + 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail = hits_at_100_tail + 1
            if ranks_tail[i] <= 10:
                hits_at_ten_tail = hits_at_ten_tail + 1
            if ranks_tail[i] <= 3:
                hits_at_three_tail = hits_at_three_tail + 1
            if ranks_tail[i] == 1:
                hits_at_one_tail = hits_at_one_tail + 1

        print(len(ranks_head), len(ranks_tail), len(
            reciprocal_ranks_head), len(reciprocal_ranks_tail))
        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        average_hits_at_100_head.append(
            hits_at_100_head / len(ranks_head))
        average_hits_at_ten_head.append(
            hits_at_ten_head / len(ranks_head))
        average_hits_at_three_head.append(
            hits_at_three_head / len(ranks_head))
        average_hits_at_one_head.append(
            hits_at_one_head / len(ranks_head))
        average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
        average_mean_recip_rank_head.append(
            sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

        average_hits_at_100_tail.append(
            hits_at_100_tail / len(ranks_head))
        average_hits_at_ten_tail.append(
            hits_at_ten_tail / len(ranks_head))
        average_hits_at_three_tail.append(
            hits_at_three_tail / len(ranks_head))
        average_hits_at_one_tail.append(
            hits_at_one_tail / len(ranks_head))
        average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
        average_mean_recip_rank_tail.append(
            sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

    print("\nAveraged stats for replacing head are -> ")
    print("Hits@100 are {}".format(
        sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
    print("Hits@10 are {}".format(
        sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
    print("Hits@3 are {}".format(
        sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
    print("Hits@1 are {}".format(
        sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
    print("Mean rank {}".format(
        sum(average_mean_rank_head) / len(average_mean_rank_head)))
    print("Mean Reciprocal Rank {}".format(
        sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

    print("\nAveraged stats for replacing tail are -> ")
    print("Hits@100 are {}".format(
        sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
    print("Hits@10 are {}".format(
        sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
    print("Hits@3 are {}".format(
        sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
    print("Hits@1 are {}".format(
        sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
    print("Mean rank {}".format(
        sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
    print("Mean Reciprocal Rank {}".format(
        sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

    cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                           + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
    cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                           + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
    cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                             + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
    cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                           + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
    cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                            + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
    cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
        average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

    print("\nCumulative stats are -> ")
    print("Hits@100 are {}".format(cumulative_hits_100))
    print("Hits@10 are {}".format(cumulative_hits_ten))
    print("Hits@3 are {}".format(cumulative_hits_three))
    print("Hits@1 are {}".format(cumulative_hits_one))
    print("Mean rank {}".format(cumulative_mean_rank))
    print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))


# %%
def load_constraint(path):
    file_path = os.path.join(path, 'constraint.txt')
    constraint = {'rel_src': {}, 'rel_dst': {}}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = [int(x) for x in line.strip().split('\t')]
            if i % 2 == 0:
                constraint['rel_src'][line[0]] = set(line[2:])
            else:
                constraint['rel_dst'][line[0]] = set(line[2:])
    return constraint

def load_n2n(path, data_file):
    file_path = os.path.join(path, data_file)
    df = pd.read_csv(file_path, sep='\t', header=None,
                     names=None, dtype=np.int64)
    df = df.drop_duplicates()
    return df.values

def evaluate(args, unique_entities):
    model_conv = Conv(num_entities, num_relations, args.embedding_size,
                      args.drop, args.alpha, args.out_channels)
    path = os.path.join(args.output_folder, 'trained_{}.pth'.format(args.epochs - 1))
    print(path)
    model_conv.load_state_dict(torch.load(path))

    print(model_conv)
    print(args.output_folder)
    model_conv.cuda()
    model_conv.eval()
    
    s11 = load_n2n(args.data, '1-1.txt')
    s1n = load_n2n(args.data, '1-n.txt')
    sn1 = load_n2n(args.data, 'n-1.txt')
    snn = load_n2n(args.data, 'n-n.txt')
    constraint = load_constraint(args.data)
    # constraint = None
    with torch.no_grad():
        print()
        print('link prediction')
        get_validation_pred(args, model_conv, test_indices, unique_entities, constraint)
        print()
        print('1 to 1 link prediction')
        get_validation_pred(args, model_conv, s11, unique_entities, constraint)
        print()
        print('1 to n link prediction')
        get_validation_pred(args, model_conv, s1n, unique_entities, constraint)
        print()
        print('n to 1 link prediction')
        get_validation_pred(args, model_conv, sn1, unique_entities, constraint)
        print()
        print('n to n link prediction')
        get_validation_pred(args, model_conv, snn, unique_entities, constraint)


# %%
# train_conv(args)
evaluate(args, set(unique_entities_train))
