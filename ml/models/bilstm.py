from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM


class BILSTMModel:

    def __init__(self, input_size):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_size, output_dim=100, input_shape=(79,)))

        self.model.add(Bidirectional(LSTM(128)))

        self.model.add(BatchNormalization())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(29, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def predict(self, data):
        return self.model.predict(data)

    def train(self, X_tr, y_tr, X_ts, y_ts, epochs=10):
        return self.model.fit(X_tr, y_tr, epochs=epochs, batch_size=32, validation_data=(X_ts, y_ts),
                              callbacks=[EarlyStopping(patience=3)])

    def save(self, path="./compiled/models/bilstm.keras"):
        self.model.save(path)
