import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
import numpy as np
import os
import visualkeras

class BIGRUModel:

    def __init__(self, input_size, num_classes=7):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_size, output_dim=100))
        self.model.add(Bidirectional(GRU(128)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        # self.model.build(input_shape=(None, input_size)) # Не є строго необхідним, якщо Embedding - перший шар

        self.input_size = input_size
        self.num_classes = num_classes
        self.dirichlet_calibrator = None

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def _get_logits(self, data):
        return self.model.predict(data)

    def train(self, X_tr, y_tr, X_ts, y_ts, epochs=10, batch_size=32):
        X_val, y_val = X_ts, y_ts
        history = self.model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
        return history

    def train_dirichlet_calibrator(self, X_calib_features, y_calib, learning_rate=0.01, epochs=50, l2_reg=1e-3):
        print("Отримання логітів для калібрувального набору...")
        calibration_logits = self._get_logits(X_calib_features)

        logit_input = Input(shape=(self.num_classes,), name='logit_input')

        calibration_layer = Dense(self.num_classes, activation=None, use_bias=True,
                                   kernel_initializer='ones',
                                   bias_initializer='zeros',
                                   kernel_regularizer=tf.keras.regularizers.L2(l2_reg),
                                   bias_regularizer=tf.keras.regularizers.L2(l2_reg),
                                   name='dirichlet_calibration_transform')
        calibrated_logits_output = calibration_layer(logit_input)

        self.dirichlet_calibrator = Model(inputs=logit_input, outputs=calibrated_logits_output, name="dirichlet_calibrator_model")

        self.dirichlet_calibrator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                           metrics=['accuracy'])

        print("Навчання калібратора Діріхле...")
        self.dirichlet_calibrator.fit(calibration_logits, y_calib,
                                       epochs=epochs,
                                       batch_size=min(32, calibration_logits.shape[0]),
                                       verbose=1)
        print("Калібратор Діріхле навчений.")


    def predict(self, data):

        logits = self._get_logits(data)

        if self.dirichlet_calibrator:
            calibrated_logits = self.dirichlet_calibrator.predict(logits)
            probabilities = tf.nn.softmax(calibrated_logits, axis=-1).numpy()
            print(probabilities)
        else:
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        return np.argmax(probabilities, axis=1)

    def save(self, path="./models/compiled/bigru.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Базова модель збережена у: {path}")


    def save_dirichlet_calibrator(self, path="./models/compiled/bigru_calibrator.keras"):

        if self.dirichlet_calibrator:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.dirichlet_calibrator.save(path)
            print(f"Калібратор Діріхле збережений у: {path}")
        else:
            print("Калібратор Діріхле не навчений. Зберігати нічого.")

    def load_model(self, path="./models/compiled/bigru.keras"):

        try:

            self.model = tf.keras.models.load_model(path, compile=False)
            self.compile()
            print(f"Базова модель завантажена з: {path}")
        except Exception as e:
            print(f"Помилка завантаження базової моделі з {path}: {e}")


    def load_dirichlet_calibrator(self, path="./models/compiled/bigru_calibrator.keras"):

        try:
            self.dirichlet_calibrator = tf.keras.models.load_model(path)
            print(f"Калібратор Діріхле завантажений з: {path}")
        except Exception as e:
            print(f"Не вдалося завантажити калібратор Діріхле з {path}: {e}. Можливо, він не був збережений або шлях невірний.")
            self.dirichlet_calibrator = None
