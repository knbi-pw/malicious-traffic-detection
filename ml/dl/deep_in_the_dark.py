import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Flatten, MaxPool1D, Dense


# def prepare_dataset(dataset, shuffle=True, shuffle_size=10000, seed=12, perc_train=0.9, perc_validation=0.1):
#
#     if shuffle:
#         dataset = dataset.shuffle(shuffle_size, seed)
#     dataset_size= len(dataset)
#     train_size = int(perc_train * dataset_size)
#     val_size = int(perc_validation * dataset_size)
#
#     train_dataset = dataset.take(train_size)
#     validation_dataset = dataset.skip(train_size).take(val_size)
#
#     return train_dataset, validation_dataset


class DeepModel:

    def __init__(self, epochs, steps_per_epoch, validation_steps):
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def model_build(self, train_dataset=None, validation_dataset=None):
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        with tf.device('/CPU:0'):
            model = Sequential()
            model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(1024,1)))
            model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
            model.add(MaxPool1D(pool_size=8))
            model.add(LSTM(units=200, return_sequences=True))
            model.add(Flatten())
            model.add(Dense(units=200, activation='relu'))
            model.add(Dense(units=200, activation='relu'))
            model.add(Dense(units=1, activation='relu'))
            model.summary()

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(train_dataset,
                                steps_per_epoch=int(self.steps_per_epoch),
                                epochs=self.epochs,
                                verbose=1,
                                validation_data= validation_dataset,
                                validation_steps=int(self.validation_steps))

        return model, history

    def model_load(self, path):
        return tf.keras.models.load_model(path)

    def model_predict(self, model, test_input):
        model.predict(test_input)
