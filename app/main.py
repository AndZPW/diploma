import sqlite3
import sys
import tensorflow as tf
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView
)
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords

DB_FILE = "history.db"
MODEL_PATH = "bigru_0-6.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100  # пришвидшує паддінг і інференс

# =================== INIT DATABASE ===================
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


# =================== LOAD TOKENIZER ===================
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


# =================== WORKER THREAD ===================
class PredictionWorker(QThread):
    result_ready = pyqtSignal(str, str)  # emotion, cleaned_text

    def __init__(self, model, tokenizer, stopwords, text):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.text = text

    def run(self):
        cleaned_text = ' '.join([w for w in self.text.split() if w.lower() not in self.stopwords])
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
        prediction = self.model.predict(padded, verbose=0)
        emotion = np.argmax(prediction, axis=1)[0]

        # Тут можна змінити на свою мапу
        emotion_label = ["радість", "сум", "гнів", "здивування", "страх", "любов"][emotion % 6]

        self.result_ready.emit(emotion_label, cleaned_text)


# =================== MAIN WIDGET ===================
class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Аналізатор емоцій")
        self.setGeometry(200, 200, 600, 500)
        self.setStyleSheet(self.load_styles())

        self.stop = stopwords.words('english')
        self.tokenizer = load_tokenizer()
        self.model = tf.keras.models.load_model(MODEL_PATH)

        layout = QVBoxLayout()

        self.label = QLabel("🔍 Введіть текст:")
        self.label.setFont(QFont("Arial", 12))
        layout.addWidget(self.label)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Введіть текст тут...")
        layout.addWidget(self.text_input)

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

        self.load_history()

    def analyze_text(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.result_label.setText("⚠ Введіть текст")
            return

        self.result_label.setText("🔄 Аналіз...")
        self.analyze_button.setEnabled(False)

        self.worker = PredictionWorker(self.model, self.tokenizer, self.stop, text)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.start()

    def handle_result(self, emotion, cleaned_text):
        self.result_label.setText(f"📌 Результат: {emotion}")
        self.save_to_history(cleaned_text, emotion, "Модель 1")
        self.load_history()
        self.analyze_button.setEnabled(True)

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


# =================== ENTRY POINT ===================
if __name__ == "__main__":
    init_db()
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec())
