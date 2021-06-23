import os
import sys
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import random

def muti_icv_protein():
    target_set = []
    with open("/home/databank/yls/DeepPurpose_DeepPurpose_outtest_2/out_test/drug_protein_out.txt") as f:
        f1 = f.readlines()
    for line in f1:
        target_set.append(line.split(",")[1].strip())
    target_set = set(target_set)
    #print(target_set)
    dti_mutidict = defaultdict(list)
    [dti_mutidict[i.split(",")[0]].append(i.split(",")[1].strip()) for i in f1]
    #print(dti_mutidict)
    whole_positive = []
    whole_negetive = []
    for key in dti_mutidict:
        #print(key)
        num = 0
        for i in dti_mutidict[key]:
            num += 1
            # whole_positive.append([key,i,dict_icv_drug_encoding[key],dict_target_protein_encoding[i],1])
            whole_positive.append([key,i,1])
        target_no = list(target_set)[:]
        [target_no.remove(z) for z in dti_mutidict[key]]
        suiji = np.random.randint(1, len(target_no), size = num)
        for a in suiji:
            #print(a)
            # whole_negetive.append([key,a,dict_icv_drug_encoding[key],dict_target_protein_encoding[a],0])
            whole_negetive.append([key,target_no[a],0])
    whole_positive = np.array(whole_positive,dtype=object)
    whole_negetive = np.array(whole_negetive,dtype=object)
    data_set = np.vstack((whole_positive,whole_negetive))
    # kf = StratifiedKFold(n_splits=100, shuffle=True, random_state=10)
    # y = LabelEncoder().fit_transform(data_set[:, 2])
    # flag = 0
    for flag in range(2):
        #train_data, test_data = data_set[train], data_set[test]
        if not os.path.exists('./test_' + str(flag)):
            os.makedirs('./test_' + str(flag))
        with open('./test_' + str(flag) + "/test.txt", "w") as rt:
            li = list(range(len(data_set)))
            random.shuffle(li)
            for tmp in li:
                rt.writelines([str(line)+',' for line in data_set[tmp]])
                rt.write("\n")
        # with open('./fold_' + str(flag) + "/val.txt", "w") as rt:
        #     li = list(range(len(test_data)))
        #     random.shuffle(li)
        #     for tmp in li:
        #         rt.writelines([str(line)+',' for line in test_data[tmp]])
        #         rt.write("\n")
        # one_fold_dataset = []
        # one_fold_dataset.append(train_data)
        # one_fold_dataset.append(test_data)
        # fold_dataset.append(one_fold_dataset)
        # flag += 1
        # if flag > 1:
            # break

def main():
    muti_icv_protein()

if __name__ == '__main__':
    main()
