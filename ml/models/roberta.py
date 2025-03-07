from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, EvalPrediction, Trainer
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from datasets import Dataset


class BERTModel:

    def __init__(self, num_labels=29, model_name="FacebookAI/roberta-base"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True)
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.args = None

    def compile(self, learning_rate=0.01):
        batch_size = 32
        self.args = TrainingArguments('outputs', learning_rate=learning_rate, warmup_ratio=0.1,
                                      fp16=True, evaluation_strategy="epoch", per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size * 2, weight_decay=0.01,
                                      report_to='none', save_total_limit=2
                                      )

    def train(self, X_train, y_train, X_test, y_test, epochs=3):
        self.args.num_train_epochs = epochs

        X_train_tokenized = self.tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
        X_test_tokenized = self.tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")

        train = Dataset.from_dict({
            "input_ids": X_train_tokenized["input_ids"],
            "attention_mask": X_train_tokenized["attention_mask"],
            "labels": y_train
        })

        test = Dataset.from_dict({
            "input_ids": X_test_tokenized["input_ids"],
            "attention_mask": X_test_tokenized["attention_mask"],
            "labels": y_test
        })

        trainer = Trainer(self.model, self.args, train_dataset=train, eval_dataset=test,
                          tokenizer=self.tokenizer, compute_metrics=metrics)

        trainer.train()
        trainer.evaluate()

    def summary(self):
        print(self.model)

    def save(self):
        self.model.save("./models/compiled/roberta.keras")


def metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.argmax(probs, axis=1)
    return {'accuracy': accuracy_score(p[1], y_pred)}
