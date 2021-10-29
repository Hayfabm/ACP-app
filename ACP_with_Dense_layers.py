""" 
ACP training script using bio-transformers and feature encodings 
"""
import datetime
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    concatenate,
    Input,
    Dense,
    Dropout,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from utils import create_dataset
from biotransformers import BioTransformers
from Feature_generation import AAC, DPC, CKSAAGP


# Model architecture
# first input model
input_1 = Input(shape=(1024, ))  # embedding_layer(None, 1024, )
dense11 = Dense(128, 64, activation="relu")(input_1)
drop11 = Dropout(0.2)(dense11)
dense12 = Dense (128, 64, activation= "relu")(drop11)
# second input model
input_2 = Input(shape=(20,))  # AAC_encoding_layer(None, 20, )
dense21 = Dense(128, 64, activation="relu")(input_2)
drop21 = Dropout(0.2)(dense21)
dense22 = Dense (128, 64, activation= "relu")(drop21)
# third input model
input_3 = Input(shape=(400,))  # DPC_encoding_layer(None, 400, )
dense31 = Dense(128, 64, activation="relu")(input_3)
drop31 = Dropout(0.2)(dense31)
dense32 = Dense (128, 64, activation= "relu")(drop31)
# forth input model
input_4 = Input(shape=(150, ))  # CKSAAGP_encoding_layer(None, 150,)
dense41 = Dense(128, 64, activation="relu")(input_4)
drop41 = Dropout(0.2)(dense41)
dense42 = Dense (128, 64, activation= "relu")(drop41)
# merge input models
merge = concatenate([dense12, dense22, dense32, dense42])
# interpretation model
hidden1 = Dense(10, activation="relu")(merge)
drop1 = Dropout(0.1)(hidden1)
output = Dense(2, activation="sigmoid")(drop1)
model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output)
# summarize layers
print(model.summary())
# plot graph
# plot_model(model, show_shapes=True)


if __name__ == "__main__":
    # init neptune logger
    run = neptune.init(
        project="sophiedalentour/ACP-app", tags=["embedding_layer", "feature_encoding"],
    )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # embedding and convolution parameters
    EMBEDDING_SIZE = 1024
    BIOTF_MODEL = "protbert"
    BIOTF_POOLMODE = "cls"
    BIOTF_BS = 2

    # training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 500
    SAVED_MODEL_PATH = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    TRAIN_SET = "datasets/train_data"
    TEST_SET = "datasets/test_data"

    # save parameters in neptune
    run["hyper-parameters"] = {
        "encoding_mode": "bio-transformers",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "saved_model_path": SAVED_MODEL_PATH,
        "train_set": TRAIN_SET,
        "test_set": TEST_SET,
    }

    # create train dataset
    sequences_train, labels_train = create_dataset(data_path=TRAIN_SET)

    # create test dataset
    sequences_test, labels_test = create_dataset(data_path=TEST_SET)

    # sequences embeddings with biotransformers(input_1)
    bio_trans = BioTransformers(backend=BIOTF_MODEL)

    sequences_train_embeddings = bio_trans.compute_embeddings(
        sequences_train, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ]  # (1378, 1024)

    sequences_test_embeddings = bio_trans.compute_embeddings(
        sequences_test, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ]  # (344, 1024)


    # input_2
    AAC_train = AAC(sequences_train)
    train_encoding_AAC = pd.DataFrame(AAC_train)
    train_encoding_AAC = np.array(train_encoding_AAC)  # (1378, 20)

    AAC_test = AAC(sequences_test)
    test_encoding_AAC = pd.DataFrame(AAC_test)
    test_encoding_AAC = np.array(test_encoding_AAC)  # (344, 20)
    
    # input_3
    DPC_train = DPC(sequences_train)
    train_encoding_DPC = pd.DataFrame(DPC_train)
    train_encoding_DPC = np.array(train_encoding_DPC)  # (1378, 400)

    DPC_test = DPC(sequences_test)
    test_encoding_DPC = pd.DataFrame(DPC_test)
    test_encoding_DPC = np.array(test_encoding_DPC)  # (344, 400)
    
    # input_4
    CKSAAGP_train = CKSAAGP(sequences_train)
    train_encoding_CKSAAGP = pd.DataFrame(CKSAAGP_train)
    train_encoding_CKSAAGP = np.array(train_encoding_CKSAAGP)  # (1378, 150)
    
    CKSAAGP_test = CKSAAGP(sequences_test)
    test_encoding_CKSAAGP = pd.DataFrame(CKSAAGP_test)
    test_encoding_CKSAAGP = np.array(test_encoding_CKSAAGP)  # (344, 150)
    

    # encode labels
    labels_train_encoded = to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (1378, 2)
    labels_test_encoded = to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (344, 2)

    # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    # define callbacks
    my_callbacks = [
        # ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        # EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
        ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            filepath=SAVED_MODEL_PATH,
            save_best_only=True,
        ),
        NeptuneCallback(run=run, base_namespace="metrics"),
    ]

    # fit the model
    history = model.fit(
        [
            sequences_train_embeddings,
            train_encoding_AAC,
            train_encoding_DPC,
            train_encoding_CKSAAGP,
        ],
        labels_train_encoded,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=(
            [
                sequences_test_embeddings,
                test_encoding_AAC,
                test_encoding_DPC,
                test_encoding_CKSAAGP,
            ],
            labels_test_encoded,
        ),
        callbacks=my_callbacks,
    )

    run.stop()
