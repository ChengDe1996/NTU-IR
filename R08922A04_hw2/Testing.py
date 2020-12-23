import torch
from Model import BPR, BCE
from MAP import calc_map, calc_recall
from Train_BPR import predict, output
import pickle
import argparse
  
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('outpath', help = 'path for predict_csv', default= 'submission.csv')
	parser.add_argument('--preprocess', help = 'preprocess data', default = 'preprocess.pickle')
	parser.add_argument('--model', help = 'which model', default = 'BPR')

	args = parser.parse_args()
	return args


def test(args, preprocess ):
	if args.model == 'BPR':
		model = BPR(4454, 3260, 2048).to('cpu')
		model.load_state_dict(torch.load('BPR.pt'))
	elif args.model == 'BCE':
		model = BCE(4454,3260,2048).to('cpu')
		model.load_state_dict(torch.load('BCE.pt'))

	with torch.no_grad():
		model.eval()
		result = predict(model, 50, preprocess['data_mask'])
		output(args.outpath, result)


def main():
	args = parse_args()
	with open(args.preprocess, 'rb') as infile:
		preprocess = pickle.load(infile)
	test(args, preprocess)


if __name__ == '__main__':
	main()



