import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from typing import Dict, List, Tuple, Union, Optional
import tensorflow as tf
import pickle
# from keras.preprocessing.sequence import pad_sequences # tf.keras.preprocessing.sequence.pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
from tqdm import tqdm

class EmotionEnsembleClassifier:

    def __init__(self,
                 model1_path: str = "SamLowe/roberta-base-go_emotions-onnx",
                 model2_path: str = "./models/compiled/bigru_0-6.keras",
                 model3_path: str = "./models/compiled/bigru_7-13.keras",
                 tokenizer2_path: str = "./tokenizer.pkl",
                 tokenizer3_path: str = "./tokenizer.pkl",
                 meta_learner_path: Optional[str] = None,
                 device: str = None):

        self.device = device if device else ('cpu')
        print(f"Using device: {self.device}")

        self.ordered_emotion_map = {
            'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3,
            'neutral': 4, 'love': 5, 'surprise': 6, 'admiration': 7,
            'approval': 8, 'amusement': 9, 'gratitude': 10, 'disapproval': 11,
            'curiosity': 12, 'annoyance': 13
        }
        self.emotion_label_map = {v: k for k, v in self.ordered_emotion_map.items()}
        self.num_emotions = len(self.ordered_emotion_map)

        self.model2_emotion_map = {
            'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3,
            'neutral': 4, 'love': 5, 'surprise': 6
        }
        self.model3_emotion_map = {
            'admiration': 0, 'approval': 1, 'amusement': 2, 'gratitude': 3,
            'disapproval': 4, 'curiosity': 5, 'annoyance': 6
        }
        self.model2_label_map = {v: k for k, v in self.model2_emotion_map.items()}
        self.model3_label_map = {v: k for k, v in self.model3_emotion_map.items()}

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.ordered_emotion_map.keys()))

        print(f"Loading model 1 from {model1_path}...")
        self.tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
        self.model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)
        self.model1.to(self.device)
        self.model1.eval()

        print(f"Loading model 2 from {model2_path}...")
        self.model2 = tf.keras.models.load_model(model2_path)
        print(f"Loading model 3 from {model3_path}...")
        self.model3 = tf.keras.models.load_model(model3_path)

        with open(tokenizer2_path, 'rb') as f:
            self.tokenizer2 = pickle.load(f)
        with open(tokenizer3_path, 'rb') as f:
            self.tokenizer3 = pickle.load(f)


        self.keras_maxlen = 128

        self.meta_learner = None
        if meta_learner_path and os.path.exists(meta_learner_path):
            self.load_meta_learner(meta_learner_path)
        else:
            print("Meta-learner not loaded. Please train or load it before prediction.")

            self.meta_learner = LogisticRegression(max_iter=1000, solver='liblinear')

    def _process_with_model(self,
                            model_name: str,
                            text: str,
                            max_length: int = 128) -> torch.Tensor:

        global_probabilities = torch.zeros(self.num_emotions, device=self.device)

        if model_name == "model1":
            inputs = self.tokenizer1(text,
                                     return_tensors="pt",
                                     truncation=True,
                                     padding="max_length",
                                     max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model1(**inputs)
                logits = outputs.logits
                local_probabilities = F.softmax(logits, dim=1).squeeze()
                global_probabilities = local_probabilities

                if local_probabilities.shape[0] == 28:
                    model1_id2label = self.model1.config.id2label
                    temp_probs = torch.zeros(self.num_emotions, device=self.device)
                    for target_idx, target_emotion_name in self.emotion_label_map.items():
                        for m1_idx, m1_emotion_name in model1_id2label.items():
                            if m1_emotion_name == target_emotion_name:
                                temp_probs[target_idx] = local_probabilities[m1_idx]
                                break
                    global_probabilities = temp_probs

                elif local_probabilities.shape[0] == self.num_emotions:
                    global_probabilities = local_probabilities
                else:
                    raise ValueError(
                        f"Model1 output size {local_probabilities.shape[0]} does not match expected {self.num_emotions} or 28 for GoEmotions.")


        elif model_name == "model2" or model_name == "model3":
            tokenizer = self.tokenizer2 if model_name == "model2" else self.tokenizer3
            model_keras = self.model2 if model_name == "model2" else self.model3
            label_map_keras = self.model2_label_map if model_name == "model2" else self.model3_label_map

            sequences = tokenizer.texts_to_sequences([text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.keras_maxlen, padding='post',
                                                                   truncating='post')

            local_probabilities_np = model_keras.predict(padded, verbose=0)[0]
            local_probabilities = torch.tensor(local_probabilities_np, device=self.device, dtype=torch.float32)

            for local_idx, emotion_name in label_map_keras.items():
                if emotion_name in self.ordered_emotion_map:  # Перевірка, чи є емоція в загальній мапі
                    global_idx = self.ordered_emotion_map[emotion_name]
                    global_probabilities[global_idx] = local_probabilities[local_idx]
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        return global_probabilities

    def _get_base_models_concatenated_outputs(self, text: str) -> np.ndarray:

        probs1 = self._process_with_model("model1", text).cpu().numpy()
        probs2 = self._process_with_model("model2", text).cpu().numpy()
        probs3 = self._process_with_model("model3", text).cpu().numpy()

        assert probs1.shape == (self.num_emotions,), f"Probs1 shape: {probs1.shape}"
        assert probs2.shape == (self.num_emotions,), f"Probs2 shape: {probs2.shape}"
        assert probs3.shape == (self.num_emotions,), f"Probs3 shape: {probs3.shape}"

        return np.concatenate([probs1, probs2, probs3])

    def train_meta_learner(self,
                           texts: List[str],
                           true_emotion_names: List[str],
                           logistic_regression_args: Optional[Dict] = None,
                           meta_learner_save_path: Optional[str] = None):

        print("Preparing features for meta-learner training...")
        meta_features = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"Processing text {i + 1}/{len(texts)} for meta-learner training...")
            concatenated_outputs = self._get_base_models_concatenated_outputs(text)
            meta_features.append(concatenated_outputs)

        X_meta = np.array(meta_features)

        y_meta = self.label_encoder.transform(true_emotion_names)

        print(f"Training meta-learner with {X_meta.shape[0]} samples and {X_meta.shape[1]} features...")

        if logistic_regression_args is None:
            logistic_regression_args = {'max_iter': 1000, 'solver': 'liblinear', 'C': 1.0,
                                        'multi_class': 'ovr'}

        self.meta_learner = LogisticRegression(**logistic_regression_args)
        self.meta_learner.fit(X_meta, y_meta)

        print("Meta-learner training complete.")

        if meta_learner_save_path:
            self.save_meta_learner(meta_learner_save_path)

    def predict(self,
                text: str,
                return_probabilities: bool = False) -> Union[str, Dict[str, float]]:

        if self.meta_learner is None or not hasattr(self.meta_learner, 'coef_'):
            is_fitted = hasattr(self.meta_learner, 'classes_') and len(self.meta_learner.classes_) > 0
            if not is_fitted:
                raise RuntimeError(
                    "Meta-learner is not trained or loaded properly. Please call train_meta_learner() or load_meta_learner().")

        concatenated_outputs = self._get_base_models_concatenated_outputs(text).reshape(1, -1)

        if return_probabilities:
            probabilities_from_lr = self.meta_learner.predict_proba(concatenated_outputs)[0]

            full_probabilities_array = np.zeros(self.num_emotions, dtype=float)

            for i, class_index_in_map in enumerate(self.meta_learner.classes_):
                full_probabilities_array[class_index_in_map] = probabilities_from_lr[i]

            final_probs = {
                self.emotion_label_map[k]: float(full_probabilities_array[k])
                for k in range(self.num_emotions)
            }
            return final_probs
        else:

            predicted_idx = self.meta_learner.predict(concatenated_outputs)[0]
            return self.emotion_label_map[predicted_idx]

    def batch_predict(self,
                      texts: List[str],
                      return_probabilities: bool = False) -> List[Union[str, Dict[str, float]]]:
        return [self.predict(text, return_probabilities) for text in texts]

    def save_meta_learner(self, file_path: str):
        """Зберігає навчену модель логістичної регресії."""
        if self.meta_learner:
            joblib.dump(self.meta_learner, file_path)
            print(f"Meta-learner saved to {file_path}")
        else:
            print("No meta-learner to save.")

    def load_meta_learner(self, file_path: str):

        try:
            self.meta_learner = joblib.load(file_path)
            print(f"Meta-learner loaded from {file_path}")
            # Перевірка, чи модель завантажена і "навчена" (має атрибут coef_)
            if not hasattr(self.meta_learner, 'coef_'):
                print("Warning: Loaded meta-learner appears to be un-trained (missing 'coef_' attribute).")
                print("Re-initializing an empty LogisticRegression model.")
                self.meta_learner = LogisticRegression(max_iter=1000, solver='liblinear')
        except FileNotFoundError:
            print(f"Error: Meta-learner file not found at {file_path}. Initializing a new one.")
            self.meta_learner = LogisticRegression(max_iter=1000, solver='liblinear')
        except Exception as e:
            print(f"Error loading meta-learner: {e}. Initializing a new one.")
            self.meta_learner = LogisticRegression(max_iter=1000, solver='liblinear')




LABELS = ['joy', 'sadness', 'anger', 'fear',
          'neutral', 'love', 'surprise', 'admiration',
          'approval', 'amusement', 'gratitude', 'disapproval',
          'curiosity', 'annoyance']


def evaluate_ensemble_on_sample(df_sampled, ensemble):
    y_true = []
    y_pred = []

    for text, label in tqdm(zip(df_sampled["text"], df_sampled["label"]), total=len(df_sampled)):
        prediction = ensemble.predict(text)
        y_true.append(label)
        y_pred.append(prediction)

    return np.array(y_true), np.array(y_pred)


def plot_accuracy_per_label(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracies = [report[str(label)]["precision"] for label in LABELS]
    print(accuracies)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=LABELS, y=accuracies, palette="viridis")
    plt.xlabel("Label (Emotion ID)")
    plt.ylabel("Precision")
    plt.title("Precision per Emotion Label (0–13)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()


if __name__ == "__main__":

    model2_path = "./models/compiled-2/bigru_0-6.keras"
    model3_path = "./models/compiled-2/bigru_7-13.keras"
    tokenizer_path = "./tokenizer.pkl"

    ensemble = EmotionEnsembleClassifier(
        model1_path="SamLowe/roberta-base-go_emotions",
        model2_path=model2_path,
        model3_path=model3_path,
        tokenizer2_path=tokenizer_path,
        tokenizer3_path=tokenizer_path,
        meta_learner_path="meta_learner.joblib"
    )


    df = pd.read_csv("./data/result_dataset.csv")


    assert {'text', 'label'}.issubset(df.columns)

    df_sampled = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=0.0001, random_state=42)).reset_index(drop=True)


    y_true, y_pred = evaluate_ensemble_on_sample(df_sampled, ensemble)

    y_true = [LABELS[i] for i in y_true]

    print(y_true, y_pred)

    print(classification_report(y_true, y_pred, digits=3))


    plot_accuracy_per_label(y_true, y_pred)



