"""ACP training script for custom model with one hot encoding"""
import datetime
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import(
    Activation,
    InputLayer,
    Dense,
    Dropout,
    Flatten,
    Convolution1D,
    MaxPooling1D
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


# Model architecture
def build_model(embedding_size, pool_length):
    """ANN_ACP model """
    custom_model = Sequential(name="ANN_ACP")
    custom_model.add(InputLayer(input_shape=((embedding_size, 1)))) 
    # CNN_input  (None, 1024, 1)
    # CNN_output (None, 1024, 1)  
    custom_model.add(Dropout(0.1))
    custom_model.add(
        Convolution1D(
            32,
            16,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="random_uniform",
            name="convolution_1d_layer1"
        )
    )
    custom_model.add(MaxPooling1D(pool_size=pool_length))
    custom_model.add(Dropout(0.2))
    custom_model.add(Dense(64, name="full_connect"))
    custom_model.add(Flatten())
    custom_model.add(Dense(2, activation="softmax"))

    return custom_model



if __name__ == "__main__":
    # init neptune logger
    run = neptune.init(
        project='sophiedalentour/ACP-app',
        tags=['Dropout_0.1'],
    )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # embedding and convolution parameters
    EMBEDDING_SIZE= 1024
    BIOTF_MODEL = "protbert"
    BIOTF_POOLMODE = "mean"
    BIOTF_BS = 2
    POOL_LENGTH = 8

    # training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    SAVED_MODEL_PATH = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    TRAIN_SET = "datasets/train_data"
    TEST_SET = "datasets/test_data"

    # save parameters in neptune
    run["hyper-parameters"] = {
        "encoding_mode": "bio-transformers",
        "seed": SEED,
        "pool_length": POOL_LENGTH,
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

    # sequences embeddings with biotransformers
    bio_trans = BioTransformers(backend=BIOTF_MODEL)

    sequences_train_embeddings = bio_trans.compute_embeddings(
        sequences_train, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ]  #(1378, 1024)
  
    sequences_test_embeddings = bio_trans.compute_embeddings(
        sequences_test, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ]  #(344, 1024)

    # sequences reshape 
    sequences_train_embeddings = sequences_train_embeddings.reshape(
        sequences_train_embeddings.shape[0], 1024, 1
    )  #(1378, 1024,1)
    sequences_test_embeddings = sequences_test_embeddings.reshape(
        sequences_test_embeddings.shape[0], 1024, 1
    )  #(344, 1024,1)

    # encode labels
    labels_train_encoded = to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (14189, 2)
    labels_test_encoded = to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (2272, 2)

    # build model
    model = build_model(EMBEDDING_SIZE, POOL_LENGTH)
    print(model.summary())

    # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    # define callbacks
    my_callbacks = [
        #ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        #EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
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
        sequences_train_embeddings,
        labels_train_encoded,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=(sequences_test_embeddings, labels_test_encoded),
        callbacks=my_callbacks,
    )

    run.stop()