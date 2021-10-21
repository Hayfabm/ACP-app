
import pandas as pd
import re
from typing import List, Tuple

# define input sequences
def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    dataset = pd.read_csv(data_path)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(dataset["sequence"])

# Amino acid composition
def aac_gen(seq,option,x,y):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.upper()
    aac=[]
    if option=='Normal':
        seq=seq
    elif option=='N':
        seq=seq[0:x]
    elif option=='C':
        seq=seq[-x:][::-1]
    elif option=='NC':
        seq=seq[0:x]+seq[-y:][::-1]
    for i in std:
        counter = seq.count(i) 
        aac+=[((counter*1.0)/len(seq))*100]
    return aac            


#Dipeptide composition
def dpc_gen(seq,option,x,y):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    seq=seq.upper()
    dpc=[]
    if option=='Normal':
        seq=seq
    elif option=='N':
        seq=seq[0:x]
    elif option=='C':
        seq=seq[-x:0][::-1]
    elif option=='NC':
        seq=seq[0:x]+seq[-y:][::-1]
    for j in std:
        for k in std:
            temp  = j+k
            count = seq.count(temp)
            dpc+=[((count*1.0)/(len(seq)-1))*100]
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