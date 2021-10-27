""" Features generation from primary sequences """
import pandas as pd
import numpy as np
from typing import List, Tuple
from utils import create_dataset


# Amino acid composition
def aac_gen(seq):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    
    aac = []
    res = []
    for i in range(len(seq)):
       
        print("***************************",i)

        sequences= seq[i]
        print("**********************my sequence",sequences)
        
        for j in std:
            counter = sequences.count(j)
            aac+=[((counter*1.0)/len(sequences))*100]
            res = aac
        return res
            
    


#Dipeptide composition
def dpc_gen(seq):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    dpc=[]
    for i in range(len(seq)):
        sequences = seq[i]
        for j in std:
            for k in std:
                temp  = j+k
                count = sequences.count(temp)
                dpc+=[((count*1.0)/(len(sequences)-1))*100]
    return dpc


# Physicochemical propreties classification 
def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair



def CKSAAGP(seq:str, gap = 5, **kw):

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in seq:
        sequences = seq[1]
        code =[]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequences)):
                p2 = p1 + g + 1
                if p2 < len(sequences) and sequences[p1] in AA and sequences[p2] in AA:
                    gPair[index[sequences[p1]]+'.'+index[sequences[p2]]] = gPair[index[sequences[p1]]+'.'+index[sequences[p2]]] + 1 
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings


# output_generation (examples)
TRAIN_SET = "datasets/train_data"
TEST_SET = "datasets/test_data"

sequences_train, labels_train = create_dataset(data_path=TRAIN_SET)
sequences_test, labels_test = create_dataset(data_path=TEST_SET)


#AAC_Train= aac_gen(sequences_train)
AAC_Test= aac_gen(sequences_test)
#print (AAC_Train)
print (AAC_Test)
#DPC_Train= dpc_gen(sequences_train)
#DPC_Test= dpc_gen(np.array(sequences_test))
#print (DPC_Train)
#print (DPC_Test)

#CKSAAGP_train= CKSAAGP(sequences_train)
#CKSAAGP_test= CKSAAGP(sequences_test)
#print (CKSAAGP_test)


