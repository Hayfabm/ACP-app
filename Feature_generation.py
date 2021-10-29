""" Features generation from primary sequences """

from typing import List, Tuple
from collections import Counter


# Amino acid composition
def AAC(seq, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    #header = ['#']
    #for i in AA:
    	#header.append(i)
    #encodings.append(header)

    for i in range(len(seq)):
        sequence = seq[i]
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings


# Dipeptide composition
def DPC(seq, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	encodings = []
	#diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
	#header = ['#'] + diPeptides
	#encodings.append(header)

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i

	for i in range(len(seq)):
		sequence = seq[i]
		code = []
		tmpCode = [0] * 400
		for j in range(len(sequence) - 2 + 1):
			tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
		if sum(tmpCode) != 0:
			tmpCode = [i/sum(tmpCode) for i in tmpCode]
		code = code + tmpCode
		encodings.append(code)
	return encodings


# Physicochemical propreties classification
def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def CKSAAGP(seq: str, gap=5, **kw):
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
    #for g in range(gap + 1):
        #for p in gPairIndex:
            #header.append(p+'.gap'+str(g))
    #encodings.append(header)

    for i in range(len(seq)):
        sequences = seq[i]
        code = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequences)):
                p2 = p1 + g + 1
                if p2 < len(sequences) and sequences[p1] in AA and sequences[p2] in AA:
                    gPair[index[sequences[p1]]+'.'+index[sequences[p2]]
                    ] = gPair[index[sequences[p1]]+'.'+index[sequences[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings
