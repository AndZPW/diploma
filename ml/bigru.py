import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import warnings
import models
import pickle
import os

# emotion_map = {
#     0: 'sadness',
#     1: 'joy',
#     2: 'love',
#     3: 'anger',
#     4: 'fear',
#     5: 'surprise',
#     6: 'admiration', 7: 'amusement', 8: 'annoyance', 9: 'approval',
#     10: 'caring',
#     11: 'confusion', 12: 'curiosity', 13: 'desire', 14: 'disappointment', 15: 'disapproval',
#     16: 'disgust', 17: 'embarrassment', 18: 'excitement', 19: 'gratitude', 20: 'grief',
#     21: 'nervousness', 22: 'optimism', 23: 'pride', 24: 'realization',
#     25: 'relief', 26: 'remorse', 27: 'neutral'
#     # Add more mappings as needed
# }


os.putenv("TF_ENABLE_ONEDNN_OPTS", "0")

DATASET_PATH = './data/filtered_dataset.csv '

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv(DATASET_PATH)

df = df.drop_duplicates()

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
}


def remove_chat_words(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in chat_words:
            words[i] = chat_words[word.lower()]
    return ' '.join(words)


df['text'] = df['text'].apply(remove_chat_words)
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

stop = stopwords.words('english')

df["text"] = df['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop])
)

df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)
# df = df[:1000]
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    df['text'],
    df['label'],
    test_size=0.3,
    random_state=42
)

X_test_raw, X_calc_raw, y_test_raw, y_calc_raw = train_test_split(
    X_test_raw,
    y_test_raw,
    test_size=0.5,
    random_state=42
)


def _tokenizer(x_train, x_test, x_calc):
    tokenizer = Tokenizer(num_words=50200)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)
    tokenizer.fit_on_texts(x_calc)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_calc = tokenizer.texts_to_sequences(x_calc)
    max_l = max(len(tokens) for tokens in x_train)
    print(max_l)
    return tensorflow.convert_to_tensor(
        pad_sequences(x_train, maxlen=max_l, padding='post'), tensorflow.int32
    ), tensorflow.convert_to_tensor(
        pad_sequences(x_test, maxlen=max_l, padding='post'), tensorflow.int32
    ), tensorflow.convert_to_tensor(
        pad_sequences(x_calc, maxlen=max_l, padding='post'), tensorflow.int32
    )


X_train, X_test, X_calc = _tokenizer(X_train_raw, X_test_raw, X_calc_raw)

y_train = tensorflow.convert_to_tensor(to_categorical(y_train_raw, num_classes=7), tensorflow.int32)
y_test = tensorflow.convert_to_tensor(to_categorical(y_test_raw, num_classes=7), tensorflow.int32)
y_calc = tensorflow.convert_to_tensor(to_categorical(y_calc_raw, num_classes=7), tensorflow.int32)

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
y_calc = np.argmax(y_calc, axis=1)

input_size = 50200 + 1
print(input_size)

from sklearn.metrics import classification_report

ordered_emotion_map = {
    'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3,
    'neutral': 4, 'love': 5, 'surprise': 6
    , 'admiration': 7,
    'approval': 8, 'amusement': 9, 'gratitude': 10, 'disapproval': 11,
    'curiosity': 12, 'annoyance': 13
}
import matplotlib.pyplot as plt

reverse_emotion_map = {v: k for k, v in ordered_emotion_map.items()}

model = models.BIGRUModel(input_size)

model.compile()
model.summary()

model.train(X_train, y_train, X_test, y_test, epochs=3)
model.train_dirichlet_calibrator(X_calc, y_calc, epochs=30)
y_pred = model.predict(X_test)

model.save(path="./models/compiled-2/bigru_0-6.keras")



print(y_test)
print(y_pred)
print(y_test.shape)
print(y_pred.shape)
report = classification_report(y_test, y_pred,
                               target_names=[reverse_emotion_map[i] for i in range(len(ordered_emotion_map))],
                               output_dict=True)

emotion_scores = {emotion: metrics['f1-score'] for emotion, metrics in report.items() if emotion in ordered_emotion_map}
print(emotion_scores)

plt.figure(figsize=(12, 6))
plt.bar(emotion_scores.keys(), emotion_scores.values(), color='skyblue')
plt.title(f'F1-Score по кожній емоції для {model.__class__.__name__}')
plt.xlabel('Емоція')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
