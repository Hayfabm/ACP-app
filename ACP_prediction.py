"""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import Dict, List
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from biotransformers import BioTransformers
from Feature_generation import AAC, DPC, CKSAAGP

Score = Dict[str, float]
ScoreList = List[Score]

# bio-transformers parameters
BIOTF_MODEL = "protbert"
BIOTF_POOLMODE = "cls"
BIOTF_BS = 2


class ACPApp:
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1

        # NOTE: if you have issues at this step, please use h5py 2.10.0
        # by running the following command: pip install h5py==2.10.0
        self.model = load_model("checkpoint/model_20211028135704.hdf5")
        print(self.model.summary())

    @staticmethod
    def score_names() -> List[str]:
        return ["ACP_prediction"]

    def compute_scores(self, sequences_list: List[str]) -> ScoreList:
        scores_list = []
        for sequence in sequences_list:
            # sequence embeddings
            self.bio_trans = BioTransformers(backend=BIOTF_MODEL)
            sequences_embeddings = self.bio_trans.compute_embeddings(
                sequences_list, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
            )[BIOTF_POOLMODE]
            sequences_embeddings = sequences_embeddings.reshape(
                sequences_embeddings.shape[0], 1024, 1
            )
            # Feature encodings
            AAC_sequence = AAC(sequences_list)
            encoding_AAC = pd.DataFrame(AAC_sequence)
            encoding_AAC = np.array(encoding_AAC)
            encoding_AAC = encoding_AAC.reshape(encoding_AAC.shape[0], 20, 1)
            DPC_sequence = DPC(sequences_list)
            encoding_DPC = pd.DataFrame(DPC_sequence)
            encoding_DPC = np.array(encoding_DPC)
            encoding_DPC = encoding_DPC.reshape(encoding_DPC.shape[0], 400, 1)
            CKSAAGP_sequence = CKSAAGP(sequences_list)
            encoding_CKSAAGP = pd.DataFrame(CKSAAGP_sequence)
            encoding_CKSAAGP = np.array(encoding_CKSAAGP)
            encoding_CKSAAGP = encoding_CKSAAGP.reshape(
                encoding_CKSAAGP.shape[0], 150, 1
            )

            # forward pass throught the model
            model_output = self.model.predict(
                [sequences_embeddings, encodding_AAC, encodding_DPC, encodding_CKSAAGP]
            )
            prob_list = [{self.score_names()[0]: prob[1]} for prob in model_output]

            return prob_list


if __name__ == "__main__":

    sequence = ["MKTVRQERLKSIVRILERSKEPVSGAQ", "KALEE", "LAGYNIVATPRGYVLAGG"]
    sequence = [
        "ATCDLLSAFGVGHAACAAHCIGHGYRGGYCNSKAVCTCRR",
        "AALKGCWTKSIPPKPCFGKR",
        "FLSLIPHAINAVGVHAKHF" "KWKLFKKIPKFLHL",
        "GLLSVLGSVVKHVIPHVVPVIAEHL",
        "GIGKFLKKAKKFGKAFVKILKK",
        "GIPCGESCVFIPCITAAIGCSCKSKVCYRN" "LIAHNQVRQV",
        "FLPVIAGVAANFLPKLFCAISKKC",
        "GWRTLLKKAEVKTVGKLALKHYL",
        "FLPAIFRMAAKVVPTIICSITKKC",
        "FAKLF",
        "ICLRLPGC",
        "GLWSKIKEAAKAAGKAALNAVTGLVNQGDQPS",
        "FLSLIPHIVSGVASIAKHF",
        "PAWRKARRWAWRMKKLAA",
        "ATRVVYCNRRSGSVVGGDDTVYYEG",
        "FAKKLAKKLAKAAL",
        "KWKLF",
        "GIGTKILGGVKTALKGALKELASTYAN",
        "YHWYGYTPQNVIGGGKLLLKLLKKLLKLLKKK",
        "KTKLFKKFAKKLAKKLKKLAKKL",
        "FAFAKIIAKIAKKII",
        "KLLKLLLKLYKKLLKLL",
        "FFPIIAGMAAKVICAITKKC",
        "GIGKFLHSAKKWGKAFVGQIMNC",
    ]  # ACP from alternative dataset
    sequence = [
        "KYPDRQIVAVFQPHTFTRTIALMDDFAASLNLADEVFLTDIFSSAR",
        "CRSTAFTCAN",
        "NLDQLLIVLATEPYFSEDLLG",
        "AGSSLKTGAKKIILYIPQNYQYDTEQGNGLQDLVKAAEELGIEVQ",
        "EFINKIKASPWFKNTVIVVSSDHLAMNNTAWKYLNKQDRNNL",
        "VRCADRHNLMYSTFRTFVFRETEFIAVTAYQNEKVTELKIENNPFAKG",
        "DSILVKWASRVFFSELLEAGV",
        "ISVAWYSIHADSGYNVCDNGYKLPLIYVV",
        "DSTRRLIGDLSDGVTLIVKVVIR",
        "RNIVIALNKIELVDR",
        "AATSAAPAATETPSALPTSQAGVAAPAADPN",
        "GIGYILDRQSTRKSWTRHFVKFGEGQDEAWRQWKGIYHLSMTTCYASFIA",
        "PTEITRR",
        "HFETVRKSFDKIWDN",
        "ESHETAPLVQALLNDDWQAQWPLDAEALAPVAVMFKTHS",
    ]  # non_ACP from alternative dataset

    app = ACPApp("cpu")

    scores = app.compute_scores(sequence)
    print(scores)
