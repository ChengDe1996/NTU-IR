import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from Model import BCE
from Preprocess import BCEData
from tqdm import tqdm, trange
from MAP import calc_map, calc_recall
import time


def parse_args():
	""" Parse args. """
	parser = argparse.ArgumentParser()

	parser.add_argument('--preprocess_data', default='preprocess.pickle')
	parser.add_argument('--save_path', default = 'BCE.pt')

	parser.add_argument('--user_size', type =  int, default = 4454)
	parser.add_argument('--item_size', type = int, default = 3260 )

	parser.add_argument('--embedding_dim', type=int, default=512)

	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--epoch_size', type=int, default=100)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--weight_decay', type=float, default=5e-7)

	parser.add_argument('--out_path',default='BCE_512.csv' )
	args = parser.parse_args()
	return args


def predict(model, k, pos_mask):
	user_item_matrix = torch.mm(model.user_embd.weight, model.item_embd.weight.transpose(0, 1))
	user_item_matrix *= pos_mask
	result = torch.argsort(user_item_matrix, dim = 1, descending = True)
	return result[:,:k].tolist()

def output(path, result):
	with open(path, 'w') as outfile:
		outfile.write('UserId,ItemId\n')
		for i in range(len(result)):
			outfile.write(str(i))
			outfile.write(',')
			temp = ' '.join(map(str,result[i]))
			outfile.write(temp)
			outfile.write('\n')



def train():
	args = parse_args()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	with open(args.preprocess_data, 'rb') as infile:
		preprocess = pickle.load(infile)
	training_set = BCEData(preprocess['train_pos'], preprocess['train_neg'])
	trainer = DataLoader(training_set, batch_size = args.batch_size, shuffle = True)
	model = BCE(args.user_size, args.item_size, args.embedding_dim).to(device)
	optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	loss_func = torch.nn.BCEWithLogitsLoss()
	best = {
		'map': 0,
		'epoch': 0,
	}

	best_map = 0
	with trange(args.epoch_size) as t:
		# for epoch in tqdm(range(args.epoch_size)):
		for epoch in t:
			model.train()
			trainer.dataset.uil_sample()
			for (u, i, label) in trainer:
				user = u.to(device)
				item = i.to(device)
				label = label.to(device)
				optimizer.zero_grad()
				prediction = model(user,item)
				loss = loss_func(prediction, label.float())
				loss.backward()
				optimizer.step()
			if epoch % 5 == 0:
				with torch.no_grad():
					model.eval()
					map_ = calc_map(args, model, preprocess['pos_mask'], preprocess['valid_pos_bool'])
					if map_ > best['map']:
						best['map'] = map_
						best['epoch'] = epoch
						torch.save(model.state_dict(), args.save_path)
					recall = calc_recall(args, model, preprocess['pos_mask'], preprocess['valid_pos_bool'])
					t.set_postfix({'loss': loss.item(),
									'map': map_.item(),
									'recall': recall.item()})
	print('Best:', best)
	model.load_state_dict(torch.load(args.save_path))
	with torch.no_grad():
		model.eval()
		result = predict(model, 50, preprocess['data_mask'])
		output(args.out_path, result)

if __name__ == '__main__':
    train()
