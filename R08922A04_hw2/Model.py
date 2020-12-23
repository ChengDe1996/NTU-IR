import torch.nn as nn
import torch

class BCE(nn.Module):
	def __init__(self, user_size, item_size, dim):
		super().__init__()
		self.user_embd = nn.Embedding(user_size, dim)
		self.item_embd = nn.Embedding(item_size, dim)
		# self.user_bias = nn.Embedding(user_size, 1)
		# self.item_bias = nn.Embedding(item_size, 1)

		nn.init.xavier_normal_(self.user_embd.weight)
		nn.init.xavier_normal_(self.item_embd.weight)
		# self.user_bias.weight.data.fill_(0.)
		# self.item_bias.weight.data.fill_(0.)

	def forward(self, u, i):
		user = self.user_embd(u)
		item = self.item_embd(i)
		# u_bias = self.user_bias(u)
		# i_bias = self.item_bias(i)
		# prediction = u_bias + i_bias + (user * item).sum(dim = -1)
		prediction = (user * item).sum(dim = -1)
		return prediction.squeeze()


class BPR(nn.Module):
	def __init__(self, user_size, item_size, dim):
		super().__init__()
		self.user_embd = nn.Embedding(user_size, dim)
		self.item_embd = nn.Embedding(item_size, dim)
		nn.init.xavier_normal_(self.user_embd.weight)
		nn.init.xavier_normal_(self.item_embd.weight)


	def forward(self, u, i, j):
		user = self.user_embd(u)
		item_i = self.item_embd(i)
		item_j = self.item_embd(j)

		prediction_i = (user * item_i).sum(dim = -1)
		prediction_j = (user * item_j).sum(dim = -1)

		return prediction_i, prediction_j
