import sqlite3
import sys

import tensorflow as tf
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView
)
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords

DB_FILE = "history.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            emotion TEXT,
            model_name TEXT
        )
    ''')
    conn.commit()
    conn.close()


def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)


class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Аналізатор емоцій")
        self.setGeometry(200, 200, 600, 500)
        self.setStyleSheet(self.load_styles())

        self.models = {
            "Модель 1": "bigru.keras",
            "Модель 2": "bigru.keras",
            "Модель 3": "bigru.keras"
        }

        self.stop = stopwords.words('english')
        self.tokenizer = load_tokenizer()
        self.current_model = None

        layout = QVBoxLayout()

        self.label = QLabel("🔍 Введіть текст:")
        self.label.setFont(QFont("Arial", 12))
        layout.addWidget(self.label)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Введіть текст тут...")
        layout.addWidget(self.text_input)

        self.model_selector = QComboBox()
        self.model_selector.addItems(self.models.keys())
        self.model_selector.currentTextChanged.connect(self.load_model_)
        layout.addWidget(self.model_selector)

        self.analyze_button = QPushButton("🧠 Аналізувати")
        self.analyze_button.clicked.connect(self.analyze_text)
        layout.addWidget(self.analyze_button)

        self.result_label = QLabel("📌 Результат: -")
        self.result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(self.result_label)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["📜 Текст", "😊 Емоція", "⚙️ Модель"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.history_table)

        self.setLayout(layout)

        self.load_model_(self.model_selector.currentText())
        self.load_history()

    def load_model_(self, model_name):
        model_path = self.models[model_name]
        try:
            self.current_model = tf.keras.models.load_model(model_path)

            self.result_label.setText(f"✔ Модель '{model_name}' завантажено")
        except Exception as e:
            self.result_label.setText(f"❌ Помилка завантаження: {e}")

    def analyze_text(self):
        print(self.text_input)
        print(type(self.text_input))
        text = self.text_input.toPlainText().strip()
        print(text)
        if not text:
            self.result_label.setText("⚠ Будь ласка, введіть текст")
            return

        if not self.current_model:
            self.result_label.setText("⚠ Модель не завантажена")
            return
        print("before predwork")
        print(self.current_model.inputs)
        print(self.current_model.input_shape)



        text = ' '.join([word for word in text.split() if word not in self.stop])

        sequence = self.tokenizer.texts_to_sequences(text)

        padded_sequence = pad_sequences(sequence, maxlen=60000, padding="post")
        print(padded_sequence.shape)
        prediction = self.current_model.predict(padded_sequence)
        print("predwork")
        emotion = np.argmax(prediction, axis=1)
        print(emotion)
        self.result_label.setText(f"📌 Результат: {emotion}")

        self.save_to_history(text, emotion, self.model_selector.currentText())
        self.load_history()

    def get_emotion_label(self, prediction):
        emotions = ["радість", "сум", "гнів", "здивування", "страх", "любов"]
        return emotions[int(prediction.argmax())]

    def save_to_history(self, text, emotion, model_name):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO history (text, emotion, model_name) VALUES (?, ?, ?)",
                       (text, emotion, model_name))
        conn.commit()
        conn.close()

    def load_history(self):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT text, emotion, model_name FROM history ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()

        self.history_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                self.history_table.setItem(i, j, QTableWidgetItem(str(cell)))

    def load_styles(self):
        return """
            QWidget {
                background-color: #2E2E2E;
                color: #E0E0E0;
                font-size: 14px;
            }
            QLabel {
                color: #FFFFFF;
            }
            QTextEdit {
                background-color: #3E3E3E;
                color: #FFFFFF;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox {
                background-color: #3E3E3E;
                color: #FFFFFF;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QTableWidget {
                background-color: #3E3E3E;
                color: #FFFFFF;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #4E4E4E;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
        """


if __name__ == "__main__":
    init_db()
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec())
