import pandas as pd
def calc_AP(ret, ans):
    AP = 0
    hit = 0
    for i, v in enumerate(ret):
        if v in ans:
            hit += 1
            AP += hit / (i + 1)

    return AP / len(ans)
    
def score(file):
    submit = pd.read_csv(file)
    ans = pd.read_csv('queries/ans_train.csv')
    N = 10
    MAP = 0
    for i in range(N):
        sub_docs = submit.retrieved_docs[i].split()
        ans_docs = ans.retrieved_docs[i].split()

        AP = calc_AP(sub_docs, ans_docs)
        recall = len(set(sub_docs) & set(ans_docs))

        print ('#%d Recall: %d(%f), AP: %f' % (i + 1, recall, recall / len(ans_docs), AP))
        MAP += AP

    print ('MAP: %f' % (MAP / N))
