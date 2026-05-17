import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import  Conv2D, Dropout, Dense, Flatten, Input, TimeDistributed, LSTM, Rescaling
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
from keras.losses import MeanSquaredError
from keras.regularizers import l2

np.random.seed(0)

SEQ_LEN = 5

def create_sequences(samples, seq_len=SEQ_LEN):
    X_seq = []
    y_seq = []

    for i in range(len(samples) - seq_len):
        sequence = samples[i:i + seq_len]

        image_paths = [frame[0] for frame in sequence]

        steering = float(sequence[-1][3])

        X_seq.append(image_paths)
        y_seq.append(steering)

    return np.array(X_seq), np.array(y_seq)

def load_data(args):
    data_df = pd.read_csv(
        'data/driving_log.csv',
        names=[
            'center',
            'left',
            'right',
            'steering',
            'throttle',
            'brake',
            'speed'
        ]
    )

    samples = data_df.values

    filtered_samples = []

    for sample in samples:

        steering = float(sample[3])

        if abs(steering) < 0.02:
            if np.random.rand() < 0.40:
                continue

        elif abs(steering) < 0.08:
            if np.random.rand() < 0.15:
                continue

    # Keep all meaningful turns/recoveries
        filtered_samples.append(sample)

    samples = np.array(filtered_samples)

    print("Original samples:", len(data_df.values))
    print("Filtered samples:", len(samples))
    
    X, y = create_sequences(samples)

    print("Min steering:", np.min(y))
    print("Max steering:", np.max(y))
    print("Mean steering:", np.mean(y))

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        shuffle=False,
        random_state=0
    )

    return X_train, X_valid, y_train, y_valid


def build_model(args):

    seq_input = Input(shape=(SEQ_LEN, 66, 200, 3))

  

    x = TimeDistributed(
        Conv2D(24, (5,5), strides=(2,2), activation='elu')
    )(seq_input)

    x = TimeDistributed(
        Conv2D(36, (5,5), strides=(2,2), activation='elu')
    )(x)

    x = TimeDistributed(
        Conv2D(48, (5,5), strides=(2,2), activation='elu')
    )(x)

    x = TimeDistributed(
        Conv2D(64, (3,3), activation='elu')
    )(x)

    x = TimeDistributed(
        Conv2D(64, (3,3), activation='elu')
    )(x)

    x = TimeDistributed(Flatten())(x)

    x = LSTM(
    64,
    return_sequences=False,
    dropout=0.2,
    kernel_regularizer=l2(1e-4)
    )(x)

    x = Dropout(args.keep_prob)(x)

    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)

    output = Dense(1)(x)

    model = Model(inputs=seq_input, outputs=output)

    model.summary()

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.keras',
        monitor='val_loss',
        save_best_only=args.save_best_only,
        mode='min',
        verbose=1
    )

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=args.learning_rate)
    )

    X_test, y_test = next(
        batch_generator(
            args.data_dir,
            X_train,
            y_train,
            args.batch_size,
            True
        )
    )

    print("TEMPORAL BATCH SHAPE:", X_test.shape)
    
    model.fit(
        batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
        steps_per_epoch=args.samples_per_epoch // args.batch_size,
        epochs=args.nb_epoch,
        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
        validation_steps=len(X_valid) // args.batch_size,
        callbacks=[checkpoint],
        verbose=1
    )

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.3)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=20)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=6000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=24)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=2e-5)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

