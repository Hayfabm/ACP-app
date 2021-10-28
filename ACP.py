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
    LSTM,
    Dense,
    Dropout,
    Flatten,
    Convolution1D,
    MaxPooling1D,
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
input_1 = Input(shape=(1024, 1))  # embedding_layer(None, 1024, 1)
conv11 = Convolution1D(
    32,
    kernel_size=16,
    activation="relu",
    kernel_initializer="random_uniform",
    name="convolution_1d_layer1",
)(input_1)
pool11 = MaxPooling1D(pool_size=3)(conv11)
conv12 = Convolution1D(
    32,
    kernel_size=16,
    activation="relu",
    kernel_initializer="random_uniform",
    name="convolution_1d_layer2",
)(pool11)
pool12 = MaxPooling1D(pool_size=3)(conv12)
drop11 = Dropout(0.2)(pool12)
lstm11 = LSTM(100, return_sequences=False, name="lstm1")(drop11)
drop12 = Dropout(0.1)(lstm11)
flat1 = Flatten()(pool12)
# second input model
input_2 = Input(shape=(20, 1))  # AAC_encoding_layer(None, 20, 1)
conv21 = Convolution1D(16, kernel_size=4, activation="relu")(input_2)
pool21 = MaxPooling1D(pool_size=2)(conv21)
drop21 = Dropout(0.1)(pool21)
conv22 = Convolution1D(16, kernel_size=4, activation="relu")(drop21)
pool22 = MaxPooling1D(pool_size=2)(conv22)
drop22 = Dropout(0.1)(pool22)
flat2 = Flatten()(drop22)
# third input model
input_3 = Input(shape=(400, 1))  # DPC_encoding_layer(None, 400, 1)
conv31 = Convolution1D(32, kernel_size=16, activation="relu")(input_3)
pool31 = MaxPooling1D(pool_size=2)(conv31)
drop31 = Dropout(0.1)(pool31)
conv32 = Convolution1D(16, kernel_size=4, activation="relu")(drop31)
pool32 = MaxPooling1D(pool_size=2)(conv32)
flat3 = Flatten()(pool32)
# forth input model
input_4 = Input(shape=(150, 1))  # CKSAAGP_encoding_layer(None, 150,1)
conv41 = Convolution1D(32, kernel_size=16, activation="relu")(input_4)
pool41 = MaxPooling1D(pool_size=2)(conv41)
drop41 = Dropout(0.1)(pool41)
conv42 = Convolution1D(16, kernel_size=4, activation="relu")(drop41)
pool42 = MaxPooling1D(pool_size=2)(conv42)
flat4 = Flatten()(pool42)
# merge input models
merge = concatenate([flat1, flat2, flat3, flat4])
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

    # sequences reshape
    sequences_train_embeddings = sequences_train_embeddings.reshape(
        sequences_train_embeddings.shape[0], 1024, 1
    )  # (1378, 1024,1)
    sequences_test_embeddings = sequences_test_embeddings.reshape(
        sequences_test_embeddings.shape[0], 1024, 1
    )  # (344, 1024,1)

    # input_2
    AAC_train = AAC(sequences_train)
    AAC_test = AAC(sequences_test)
    train_encoding_AAC = pd.DataFrame(AAC_train)
    train_encoding_AAC = np.array(train_encoding_AAC)  # (1378, 20)
    train_encoding_AAC = train_encoding_AAC.reshape(
        train_encoding_AAC.shape[0], 20, 1
    )  # (1378, 20, 1)
    test_encoding_AAC = pd.DataFrame(AAC_test)
    test_encoding_AAC = np.array(test_encoding_AAC)  # (344, 20)
    test_encoding_AAC = test_encoding_AAC.reshape(
        test_encoding_AAC.shape[0], 20, 1
    )  # (344, 20, 1)
    # input_3
    DPC_train = DPC(sequences_train)
    DPC_test = DPC(sequences_test)
    train_encoding_DPC = pd.DataFrame(DPC_train)
    train_encoding_DPC = np.array(train_encoding_DPC)  # (1378, 400)
    train_encoding_DPC = train_encoding_DPC.reshape(
        train_encoding_DPC.shape[0], 400, 1
    )  # (1378, 400, 1)
    test_encoding_DPC = pd.DataFrame(DPC_test)
    test_encoding_DPC = np.array(test_encoding_DPC)  # (344, 400)
    test_encoding_DPC = test_encoding_DPC.reshape(
        test_encoding_DPC.shape[0], 400, 1
    )  # (344, 400, 1)
    # input_4
    CKSAAGP_train = CKSAAGP(sequences_train)
    CKSAAGP_test = CKSAAGP(sequences_test)
    train_encoding_CKSAAGP = pd.DataFrame(CKSAAGP_train)
    train_encoding_CKSAAGP = np.array(train_encoding_CKSAAGP)  # (1378, 150)
    train_encoding_CKSAAGP = train_encoding_CKSAAGP.reshape(
        train_encoding_CKSAAGP.shape[0], 150, 1
    )  # (344, 150, 1)

    test_encoding_CKSAAGP = pd.DataFrame(CKSAAGP_test)
    test_encoding_CKSAAGP = np.array(test_encoding_CKSAAGP)  # (344, 150)
    test_encoding_CKSAAGP = test_encoding_CKSAAGP.reshape(
        test_encoding_CKSAAGP.shape[0], 150, 1
    )  # (344, 150, 1)

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
