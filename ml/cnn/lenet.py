from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


class LeNetModel:

    def __init__(self, epochs, steps_per_epoch, validation_steps):
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.model = Sequential()

    def model_build(self, train_gen=None, test_gen=None):
        self.model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dense(units=2, activation='softmax'))
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        history = self.model.fit(x=train_gen,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=test_gen,
                                 validation_steps=self.validation_steps)
        return history

    def model_save(self, path):
        self.model.save(path)

    def model_load(self, path):
        self.model = keras.models.load_model(path)

    def model_predict(self, test_input):
        return self.model.predict(test_input)
