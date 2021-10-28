# ACP-app
deepchain.bio ACP Design 

## Install ACP conda environment 

From the root of this repo, run conda env create -f environment.yaml

Follow this[tutorial](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras#step-5-monitor-your-tensorflow-keras-training-in-neptune) to make neptune logger works

## Overview 

Cancer is a leading cause of death worldwide, accounting for nearly 10 million deaths in 2020. Conventional cancer treatment relies on radiotherapy and chemotherapy, but both methods bring serious side effects to patients, as these therapies not only attack cancer cells but also damage normal cells. Anticancer peptides (ACPs), as  a new type of therapeutic agent, have gained more and more attention since they have been selected as a safe drug.  Therefore, it’s necessary to develop an efficient and accurate method to predict ACPs. Here, we developed a new algorithm to predict ACPs by combining natural language processing  and different features based on deep learning.  In this work, the Protbert language model was used to extract embeddings using [bio-transformers](https://pypi.org/project/bio-transformers/). In order to make full use of the physical and chemical properties of the peptide sequence, the AAC, DPC, CKSAAGP  features were added to the inputs of the model.

## bio-transformers_embeddings :sparkles: 

The embedding of a an object is a representation of the object in a lower dimensional space. In this lower space, it is easier to manipulate, visualize, and apply mathematical functions on proteins' projection. Embeddings model will take a sequence of amino acids in input (string) and return a vector of lower dimension.

You can choose a backend and pass a list of sequences of Amino acids to compute the embeddings. By default, the compute_embeddings function returns the <CLS> token embeddings. You can add a pool_mode in addition, so you can compute the mean of the tokens embeddings.


'''python
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(backend="protbert")
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'),batch_size=2)

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
'''









