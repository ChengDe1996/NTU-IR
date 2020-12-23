import numpy as np
from numpy.linalg import norm
import time
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import jieba
import json
import pickle
from score import score

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab_all', help = 'path of vocab.all', default = './model/vocab.all')
	parser.add_argument('--inverted_file', help = 'path of inverted file', default = './model/inverted-file')
	parser.add_argument('--file_list', help = 'path of file list', default = './model/file-list')
	parser.add_argument('-i','--query_train', default = './queries/query-train.xml' )
	parser.add_argument('--query_test', default = './queries/query-test.xml')
	parser.add_argument('--ans_train', default = './queries/ans_train.csv')
	parser.add_argument('-o','--predict_file', default = 'predict_testing.csv')
	parser.add_argument('--k1', type = float, default = 1.25)
	parser.add_argument('--k2', type = float, default = 1.25)
	parser.add_argument('--b', type = float, default = 0.75)
	parser.add_argument('--top_n', type = int, default = 100)
	parser.add_argument('-m','--model_dir', help = 'model_dir', default = '')
	parser.add_argument('-d','--nctir_dir', help = 'path of corpus', default = './CIRB010/')
	parser.add_argument('-r', "--relevance", help = 'use relevance feedback', action = 'store_true')
	# parser.add_argument('--docs', default = )
	args = parser.parse_args()
	return args

def load_query(path):
	"""return a list of dictionary """
	query_list = []
	tree = ET.parse(path)
	for topic in tree.findall('topic'):
		dic = {}
		dic['number'] = topic.find('number').text
		temp = jieba.lcut(topic.find('title').text)
		tt = []
		for i in range(len(temp)):
			if(len(temp[i])>2):
				for j in range(len(temp[i])-1):
					tt.append(temp[i][j:j+2])
			else:
				tt.append(temp[i])
		dic['title'] = tt
		temp = jieba.lcut(topic.find('question').text)[1:-1]
		tt = []
		for i in range(len(temp)):
			if(len(temp[i])>2):
				for j in range(len(temp[i])-1):
					tt.append(temp[i][j:j+2])
			else:
				tt.append(temp[i])

		dic['question'] = tt

		temp = jieba.lcut(topic.find('narrative').text)[1:-1]

		tt = []
		for i in range(len(temp)):
			if(len(temp[i])>2):
				for j in range(len(temp[i])-1):
					tt.append(temp[i][j:j+2])
			else:
				tt.append(temp[i])

		dic['narrative'] = tt

		temp = jieba.lcut(topic.find('concepts').text)[1:-1]
		tt = []
		for i in range(len(temp)):
			if(len(temp[i])>2):
				for j in range(len(temp[i])-1):
					tt.append(temp[i][j:j+2])
			else:
				tt.append(temp[i])
		dic['concept'] = tt
		query_list.append(dic)
	return query_list


def make_query(queries):
	final_queries = {}
	for query in queries:
		nar = query['narrative']
		con = query['concept']
		que = query['question']
		que.extend(con)
		que.extend(nar)
		q = que
		# final_queries[query['number']] = q
		temp = []
		for i in range(len(q)-1):
			if(len(q[i])==1 and len(q[i+1]) == 1):
				temp.append(q[i]+q[i+1])
				temp.append(q[i])
			else:
				temp.append(q[i])
		final_queries[query['number']] = temp
	return final_queries

def load_inverted_file(path):
	""" retrun a dictionary that 
		{(key1, key2):{document with this term : term freq}} """
	with open(path)as infile:
		inver = infile.readlines()
	inv_dict = {}
	idx = 0
	key = (0,0)
	count = 0
	while(idx<len(inver)):
		temp = inver[idx].split()
		ts = {}
		if(len(temp) == 3):
			key = (int(temp[0]), int(temp[1]))
			count = int(temp[2])
			ts['count'] = count
			idx += 1
			temp2 = inver[idx].split()
			while(len(temp2) != 3):
				ts[temp2[0]] = int(temp2[1])
				idx += 1
				try:
					temp2 = inver[idx].split()
				except:
					break
			inv_dict[key] = ts
	return inv_dict

def load_voc(path):
	idx2word = {}
	word2idx = {}
	with open(path)as infile:
		voc_set = infile.readlines()
	for i in range(len(voc_set)):
		idx2word[str(i)] = voc_set[i].strip()
		word2idx[voc_set[i].strip()] = i
	return idx2word, word2idx

def load_doc(path):
	with open(path)as infile:
		docs = infile.readlines()
	docs_len = {}
	docs_id_to_name = {}
	docs_name_to_id = {}
	counts = 0
	for i in range(len(docs)):
		temp = ""
		p = './CIRB010/' + docs[i][:-1]
		tree = ET.parse(p)
		root = tree.getroot()
		# txt = tree.findall('text')
		for txt in root.iter('p'):
			temp = temp + txt.text
		l = len(temp.strip())
		docs_len[i] = l
		counts += l
		for txt in root.iter('id'):
			docs_id_to_name[str(i)] = txt.text
			docs_name_to_id[txt.text] = str(i)
			# docs_id.append(txt.text)
	# print(docs_name_to_id)
	# print(docs_name_to_id['cte_foc_0000260'])

	""" docs_len is a dictionary, {(idx of this document):(this document len)}"""
	""" docs_id is a dictionary, {(idx of this document):(this document's id)}"""
	return docs_len, docs_id_to_name, docs_name_to_id, counts/len(docs)



# (k+1)c(t,d)/(c(t,d)+k(1-b+b(D/avg)))
def doc_tf_norm(queries, final_queries, inv_dict, word2idx,doc_len, avg_len, k1, b, idx2word ):
	"""return a dictionary:
	{(query_number):{(term in this query):{(id of document which have this term):(term freq of this doc)}}}"""
	queries_okapi_tf = {}
	# query_idx = 0
	for query in queries:
		# if(query['number']!='CIRB010TopicZH006'):
		# 	continue
		query_dict = {}
		narrative = final_queries[query['number']]
		# print(narrative)
		for term in narrative:
			if(term[0] not in word2idx):
				continue
			key1 = word2idx[term[0]]
			if(len(term) == 1):
				key2 = -1
			else:
				if(term[1] in word2idx):
					key2 = word2idx[term[1]]
				else:
					continue
			# print(narrative[i], key1, key2)
			key = (key1, key2)
			# print(key, term)
			tf_dict = {}
			if(key in inv_dict):
				sub_dict = inv_dict[key]
				# tf_dict = {}
				# print(sub_dict)
				for k, v in sub_dict.items():
					# print(k,v)
					if(k == 'count'):
						continue
					tf_dict[k] = ((k1+1)*v)/(v+k1*(1-b+b*(doc_len[k]/avg_len)))
				query_dict[term] = tf_dict
				# print(query_dict['財政'])
				# print(query_dict)
		queries_okapi_tf[query['number']] = query_dict
		# print(queries_okapi_tf['CIRB010TopicZH006']['財政'])
		# print(query['number'], query_dict)

		# query_idx += 1
	# print(queries_okapi_tf['CIRB010TopicZH006'])
	return queries_okapi_tf

def query_tf_norm(queries, final_queries, k1):
	"""return a dictionary : 
		{[query_number]:{(term in this query) : (this's term's tf of itself)}}"""
	queries_tf = {}
	for query in queries:
		narrative = final_queries[query['number']]
		query_dict = {}
		for term in narrative:
			count = 0
			for i in range(len(narrative)):
				if(term == narrative[i]):
					count += 1
			query_dict[term] = (k1+1)*count/(count + k1)
		# if(query['number'] == 'CIRB010TopicZH006'):
		# 	pass
		queries_tf[query['number']] = query_dict

	return queries_tf


def query_idf_norm(queries, final_queries, inv_dict, word2idx):
	""" return a dictionary : 
	{['query_number']:{('term in this query'):(this term's idf)}} """
	queries_idf = {}
	for query in queries:
		narrative = final_queries[query['number']]
		query_dict = {} #save this query's every term's idf, key is term, val is idf
		for term in narrative:
			if(term[0] not in word2idx):
				continue
			key1 = word2idx[term[0]]
			if(len(term) == 1):
				key2 = -1
			else:
				if(term[1] in word2idx):
					key2 = word2idx[term[1]]
				else:
					continue

			key = (key1, key2)
			if(key in inv_dict):
				sub_dict = inv_dict[key]
				query_dict[term] = np.log((46972+0.5)/(sub_dict['count']+0.5))
		queries_idf[query['number']] = query_dict #save every query's idf, key is query number,
	# print(queries_idf['CIRB010TopicZH007'])
	# print(queries_idf['CIRB010TopicZH008'])
	# print(queries_idf['CIRB010TopicZH001'])
	return queries_idf

def all_pos_doc(queries, final_queries, inv_dict, word2idx):
	"""return a dictionary:
		{query_number: [all possibile documents' idx]"""
	all_possi_doc = {}
	for query in queries:
		collect = []
		narrative = final_queries[query['number']]

		for term in narrative:
			if(term[0] not in word2idx):
				continue
			key1 = word2idx[term[0]]
			if(len(term) == 1):
				key2 = -1
			else:
				if(term[1] in word2idx):
					key2 = word2idx[term[1]]
				else:
					continue
			key = (key1, key2)
			if(key in inv_dict):
				sub_dict = inv_dict[key]
				for k, v in sub_dict.items():
					if(k == 'count'):
						continue
					# print(k)
					collect.append(k)
		# print(len(collect))
		collect = list(set(collect))
		# print(len(collect))
		all_possi_doc[query['number']] = collect
	return all_possi_doc

def TFIDF(queries, final_queries, queries_doc_tf, queries_tf, queries_idf, all_possible_doc, word2idx, docs_id):
	"""return a dictionary
		{(query_number):{(document_idx):list[document TFIDF at every term]}}"""
	"""queries : [list of {number:(), title:(), question:(), narrative:(), concept:()}]"""
	"""queries_doc_tf : {(query_number):{(term in this query):{(id of document which have this term):(term freq of this doc)}}}"""
	"""queries_tf : {[query_number]:{(term in this query) : (this's term's tf of itself)}}"""
	"""queries_idf : {['query_number']:{('term in this query'):(this term's idf)}}"""
	tf_idf = {}
	for query in queries:
		# if(query['number'] == 'CIRB010TopicZH001'):
		# 	print(len(queries_doc_tf[query['number']]), len(queries_tf[query['number']]), len(queries_idf[query['number']]))
		query_number = query['number']
		narrative = final_queries[query_number]
		possible_doc = all_possible_doc[query_number]
		temp_dict = {}
		for doc_idx in possible_doc:
			# if(query['number'] == 'CIRB010TopicZH006' and doc_idx == '41102' ):
			# 	print('true')
			doc_vec = []
			for term in narrative:
				if(term in queries_doc_tf[query_number]):
					if(doc_idx in queries_doc_tf[query_number][term]):
						doc_vec.append(queries_doc_tf[query_number][term][doc_idx]*queries_idf[query_number][term])
					else:
						doc_vec.append(0)

			# if( docs_id[doc_idx] == 'ctc_sto_0006502' or docs_id[doc_idx] == 'cdn_soc_0003227'):
			# 	print(doc_vec)
			temp_dict[doc_idx] = np.array(doc_vec)


		query_vec = []
		for term in narrative:
			if(term in queries_idf[query_number]):
				query_vec.append(queries_tf[query_number][term]*queries_idf[query_number][term])
		temp_dict['query'] = np.array(query_vec)
		tf_idf[query_number] = temp_dict
	# print(len(tf_idf['CIRB010TopicZH006']), tf_idf['CIRB010TopicZH006']['41102'])
	# print(tf_idf['CIRB010TopicZH006']['query'])
	return tf_idf

def cos_sim(vec1, vec2):
	"""cosine similarity"""
	return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))

def VSM(queries_tf_idf, queries, docs_id, docs_name):
	"""return a dictionary
		{query_number: sorted_related_doc{doc_name:cos_sim_score}"""
	"""{(query_number):{(document_idx):list[document TFIDF vector at every term]}}"""
	idx = 0
	queries_related_doc = {}
	for query in queries:
		# doc_sim = {'doc_id':[], 'cos_sim':[]}
		# doc_sim = pd.DataFrame(doc_sim)
		doc_sim = {}
		tf_idf = queries_tf_idf[query['number']]
		# if(query['number'] == 'CIRB010TopicZH006'):
			# print('41102' in tf_idf)
		for k,v in tf_idf.items():
			if(k == 'query'):
				continue
			# if(query['number'] == 'CIRB010TopicZH006'):
				# print('query!')
				# print(docs_name['ctc_sto_0006502'], k)
				# if(k == '41102' or k == docs_name['cdn_soc_0003227']):
				# 	print('vec:', tf_idf['query'], v)
				# 	print(cos_sim(tf_idf['query'], v))
			doc_sim[docs_id[k]] = cos_sim(tf_idf['query'], v)

		doc_sim = {k: v for k, v in sorted(doc_sim.items(), key=lambda item: item[1], reverse = True)}
		# if(query['number'] == 'CIRB010TopicZH006'):
		# 	for k,v in doc_sim.items():
		# 		print(docs_name[k],v)
		# 		exit()
		queries_related_doc[query['number']] = doc_sim
	# print(queries_related_doc['CIRB010TopicZH006'])
	# print(queries_related_doc['CIRB010TopicZH002'])
	return queries_related_doc

def gen_ans(queries, queries_related_doc, train = True, top_n = 100):
	idx = 0
	ans = {}
	for query in queries:
		if(train):
			if(idx == 10):
				break

		# print(idx)
		n = 0
		ans_list = []
		for doc,v in queries_related_doc[query['number']].items():
			if(n == top_n):
				break
			ans_list.append(doc)
			n += 1
		ans[query['number'][-3:]] = ans_list

		idx += 1

	# print(ans['006'])
	# print(ans['007'])
	return ans


def VSM_pack(queries, final_queries, inv_dict, word2idx, idx2word, docs_len, avg_len, docs_id, args, docs_name):
	k1 = args.k1
	k2 = args.k2
	b = args.b
	query = queries[5]
	queries_doc_tf = doc_tf_norm(queries, final_queries, inv_dict, word2idx, docs_len, avg_len, k1, b, idx2word )
	"""{(query_number):{(term in this query):{(id of document which have this term):(term freq of this doc)}}}"""
	queries_tf = query_tf_norm(queries, final_queries, k2)
	"""{[query_number]:{(term in this query) : (this's term's tf of itself)}}"""
	queries_idf = query_idf_norm(queries, final_queries, inv_dict, word2idx)
	"""{['query_number']:{('term in this query'):(this term's idf)}}"""
	all_possible_doc = all_pos_doc(queries, final_queries, inv_dict, word2idx)
	"""{query_number: [ list of all possibile documents' idx]"""
	queries_tf_idf = TFIDF(queries, final_queries, queries_doc_tf, queries_tf, queries_idf, all_possible_doc, word2idx, docs_id)
	"""{(query_number):{(document_idx):list[document TFIDF at every term]}}"""
	queries_related_doc = VSM(queries_tf_idf, queries, docs_id, docs_name)

	ans = gen_ans(queries, queries_related_doc, train = False, top_n = args.top_n)

	return ans

def rocchio(file_list, queries, ans, docs_name, corpus_path):
	"""rocchio relevance feedback, extend query with some word in sudo positive docs"""
	with open(file_list)as infile:
		docs = infile.readlines()

	for query in queries:
		query_number = query['number'][-3:]
		rank_n = 0
		expend_data = ""
		# print(docs_name)
		for doc in ans[query_number]:
			if(rank_n == 10):
				break
			p = corpus_path + docs[int(docs_name[doc])][:-1]
			tree = ET.parse(p)
			root = tree.getroot()
			for txt in root.iter('p'):
				expend_data = expend_data + txt.text.strip()
			rank_n += 1
		corpus = jieba.lcut(expend_data) 
		term_freq = {}
		for term in corpus:
			if(term in term_freq):
				term_freq[term] += 1
			else:
				term_freq[term] = 1
		term_freq = {k: v for k, v in sorted(term_freq.items(), key=lambda item: item[1], reverse = True)}
		ten = 0
		for k,v in term_freq.items():
			if(ten == 20):
				break
			ten += 1
			query['narrative'].append(k)
	return queries

def build_output(ans, path, train = True):
	# print(ans)
	with open(path, 'a')as outfile:
		idx = 0
		outfile.write('query_id,retrieved_docs')
		outfile.write('\n')
		for key, value in ans.items():
			if(train and idx == 10):
				break
			if(not train and idx<10):
				continue
			outfile.write(key + ',')
			for doc in value:
				outfile.write(doc + ' ')
			outfile.write('\n')
		idx += 1

def main():
	time_init = time.time()
	args = parse_args()
	queries = load_query(args.query_train)
	# queries2 = load_query(args.query_test)
	# queries.extend(queries2)
	final_queries = make_query(queries)
	# print(queries)
	with open('inv_dict.pickle', 'rb')as infile:
		inv_dict = pickle.load(infile)

	# print(inv_dict[(174,-1)])

	# idx2word, word2idx = load_voc(args.vocab_all)
	# with open('idx2word.json','w')as outfile:
	# 	json.dump(idx2word, outfile)
	# with open('word2idx.json', 'w')as outfile:
	# 	json.dump(word2idx, outfile)
	with open('idx2word.json')as infile:
		idx2word = json.load(infile)
	with open('word2idx.json')as infile:
		word2idx = json.load(infile)
	# docs_len, docs_id, docs_name, avg_len = load_doc(args.file_list)
	avg_len = 766.223
	# with open('docs_len.json', 'w')as outfile:
	# 	json.dump(docs_len, outfile)
	# with open('docs_id.json', 'w')as out:
	# 	json.dump(docs_id, out)
	# with open('docs_name.json', 'w')as out:
	# 	json.dump(docs_name, out)
	# print('avg_len:', avg_len)
	avg_len = 766.223
	with open('docs_len.json')as infile:
		docs_len = json.load(infile)
	with open('docs_id.json')as infile:
		docs_id = json.load(infile)
	with open('docs_name.json') as infile:
		docs_name = json.load(infile)
	with open('inv_dict.pickle', 'rb')as infile:
		inv_dict = pickle.load(infile)

	# a,b = word2idx['財'], word2idx['政']
	# print(a,b)
	# print(inv_dict[(a,b)])
	# print(docs_id['1261'], docs_id['2036'], docs_id['1925'])


	ans = VSM_pack(queries, final_queries, inv_dict, word2idx, idx2word, docs_len, avg_len, docs_id, args, docs_name)
	if(args.relevance):
		queries2 = rocchio(args.file_list, queries, ans, docs_name, args.nctir_dir)
		ans = VSM_pack(queries2, final_queries, inv_dict, word2idx, idx2word, docs_len, avg_len, docs_id, args, docs_name)
		print('r')
	build_output(ans, args.predict_file)
	score(args.predict_file)






if __name__ == '__main__':
	main()




