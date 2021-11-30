# ACP-app
deepchain.bio | ACP Design 

## Install ACP conda environment 

From the root of this repo, ```run conda env create -f environment.yaml```

Follow this [tutorial](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras#step-5-monitor-your-tensorflow-keras-training-in-neptune) to make neptune logger works

## Overview 

Cancer is a leading cause of death worldwide, accounting for nearly 10 million deaths in 2020. Conventional cancer treatment relies on radiotherapy and chemotherapy, but both methods bring serious side effects to patients, as these therapies not only attack cancer cells but also damage normal cells. Anticancer peptides (ACPs), as  a new type of therapeutic agent, have gained more and more attention since they have been selected as a safe drug.  Therefore, it’s necessary to develop an efficient and accurate method to predict ACPs. Here, we developed a new algorithm to predict ACPs by combining natural language processing  and different features based on deep learning.  In this app, the Protbert language model was used to extract embeddings using [bio-transformers](https://pypi.org/project/bio-transformers/). In order to make full use of the physical and chemical properties of the peptide sequence, the AAC, DPC, CKSAAGP  features were added to the inputs of the model.

## Inferring embeddings using bio-transformers

To infer embeddings, you need to install __bio-transformers__ directly from PyPI release by running:

``` pip install bio-transformers```

You can choose a __Protbert__ OR __ESM backends__ from the source with:

```> from biotransformers import BioTransformers
   > BioTransformers.list_backend()
```
```>>
Use backend in this list :

    *   esm1_t34_670M_UR100
    *   esm1_t6_43M_UR50S
    *   esm1b_t33_650M_UR50S
    *   esm_msa1_t12_100M_UR50S
    *   protbert
    *   protbert_bfd
```
By listing amino-acid sequences, the compute_embeddings function returns the <CLS> token embeddings. You can add a pool_mode in addition, so you can compute the mean of the tokens embeddings.


```python
from biotransformers import BioTransformers

sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

bio_trans = BioTransformers(backend="protbert")
embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'),batch_size=2)

cls_emb = embeddings['cls']
mean_emb = embeddings['mean']
```

## features extraction and selection from peptide sequences

Structural and physiochemical descriptors extracted from protein sequences have been widely used to represent protein sequences and predict structural, functional, expression and interaction profiles of proteins and peptides as well as other macromolecules. Here, we used [iFeature](https://github.com/Superzchen/iFeature), a versatile Python-based toolkit for generating three numerical feature representation schemes __AAC__, __DPC__, __CKSAAGP__. 

## Dataset

*_train _dataset_ contains __861__ experimentally validated ACPs and __861__ non-ACPs (or AMPs) data.
*_test_Dataset_ contains __172__ experimentally validated anticancer peptides as positive peptides and __172__ AMPs but do not show anticancer activity as negative peptides.








