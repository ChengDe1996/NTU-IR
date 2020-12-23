import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset
import pickle
import random


random.seed(6174)
def parse_args():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--predict', help = 'path of my prediction', default = 'predict.csv')
	parser.add_argument('--train', help = 'path of training file', default = './wm-2020-personalized-recommendation/train.csv')
	parser.add_argument('--tt_ratio', help = 'train test ratio', type = float, default = 0.11)
	args = parser.parse_args()
	return args


def load_data(path):
	data = pd.read_csv(path).ItemId
	data = [[int(elem) for elem in row.split()] for row in data]
	max_user, max_item = find_table_size(data)

	return data, max_user, max_item

def make_train_valid(data, train_test_ratio):
	train = []
	valid = []
	for i in range(len(data)):
		train_len = int(len(data[i])*(1-train_test_ratio))
		per_data = np.random.permutation(data[i])
		train.append(sorted(per_data[:train_len]))
		valid.append(sorted(per_data[train_len:]))

	return train, valid

def find_table_size(data):
	max_u = len(data)
	max_i = max([max(row) for row in data])+1

	return max_u,max_i


def pos_neg_sep(data, max_item):
	positive = []
	negative = []
	for row in data:
		temp_pos, temp_neg = [],[]
		row_set = set()
		for e in row:
			row_set.update([e])
		for i in range(max_item):
			if i in row_set:
				temp_pos.append(i)
			else:
				temp_neg.append(i)
		positive.append(temp_pos)
		negative.append(temp_neg)
	return positive, negative

def bool_matrix(positives, item_size):
	matrix = torch.zeros(len(positives), item_size)
	for user_id in range (len(positives)):
		for item_id in positives[user_id]:
			matrix[user_id][item_id] = 1
	return matrix



class BCEData(Dataset):
	def __init__(self, pos, neg):
		super().__init__()
		self.pos = pos
		self.neg = neg
		self.data = []
		self.uil_sample()
	def uil_sample(self):
		self.data = []
		for user_id in range(len(self.pos)):
			pos_item_ids = self.pos[user_id]
			neg_item_ids = random.choices(self.neg[user_id], k = len(pos_item_ids))
			for i in range(len(pos_item_ids)):
				self.data += [[user_id, pos_item_ids[i], 1.0]]
				self.data += [[user_id, neg_item_ids[i], 0.0]]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

class BPRData(Dataset):
	def __init__(self, pos, neg):
		super().__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.pos = pos
		self.neg = neg
		self.data = []
		self.upn_sample()
		# self.features = features
		# self.num_item = num_item
		# self.train_mat = train_mat
		# self.num_ng = num_ng
		# self.is_training = is_training


	def upn_sample(self):
		"""User, positive item, negative item sampling"""
		self.data = []
		# pos: [[pos item id for user i] for i in range all users]
		for t in range(4):
			for user_id in range(len(self.pos)):
				item_ids = self.pos[user_id]
				neg = random.choices(self.neg[user_id], k=len(item_ids))
				self.data += [[user_id, item_id, neg[i]] for i, item_id in enumerate(item_ids)]




	def __len__(self):
		return len(self.data)
		# return self.num_ng * len(self.features) if \
		# 		self.is_training else len(self.features)

	def __getitem__(self, idx):
		return self.data[idx]
		# features = self.features_fill if \
		# 		self.is_training else self.features

		# user = features[idx][0]
		# item_i = features[idx][1]
		# item_j = features[idx][2] if \
		# 		self.is_training else features[idx][1]
		# return user, item_i, item_j 



if __name__ == '__main__':
	args = parse_args()
	saving = {}
	data, user_size, item_size = load_data(args.train)
	saving['data'] = data
	train, valid = make_train_valid(data,args.tt_ratio)
	saving['train'], saving['valid'] = train, valid
	# print(data)
	train_pos, train_neg = pos_neg_sep(train, item_size)
	valid_pos, valid_neg = pos_neg_sep(valid, item_size)
	saving['train_pos'], saving['train_neg'] = train_pos, train_neg
	saving['valid_pos'], saving['valid_neg'] = valid_pos, valid_neg
	saving['data_mask']  = 1 - bool_matrix(data, item_size)
	saving['pos_mask'] = 1 - bool_matrix(train_pos, item_size)
	saving['valid_pos_bool'] = bool_matrix(valid_pos, item_size)


	with open('preprocess.pickle', 'wb') as outfile:
	    pickle.dump(saving, outfile, protocol=pickle.HIGHEST_PROTOCOL)

