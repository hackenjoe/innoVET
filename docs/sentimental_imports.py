# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import nltk
import re
import itertools
import ipywidgets as widgets
from nltk.corpus import stopwords
#from pandas_profiling import ProfileReport
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import torch
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import nltk
import re
import itertools
import ipywidgets as widgets
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm, tqdm_notebook
from germansentiment import SentimentModel
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List
from IPython.core.display import display, HTML
from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
nltk.download('stopwords')
tqdm.pandas()
pd.set_option('max_colwidth', 200)

def bewertungsfunktion(df, model_b, n=10):
    """Testfunktion für die Auswertung neuer zufälliger Handwerkerbewertungen"""
    to_emoji = lambda x: '😄' if 'positive' in x else ('😡' if 'neutral' not in x else '😐')
    rows = []
    for _ in range(n):
        with torch.no_grad():
            actual, text = random.choice(list(zip(df.Ranking, df.Bewertungstext)))
            predicted = to_emoji(model_b.predict_sentiment([text])[0])
            actual = '😄' if actual in [4, 5] else ( '😡' if actual in [1, 2] else '😐')
            row = f"<tr><td>{text}&nbsp;</td><td>{predicted}&nbsp;</td><td>{actual}&nbsp;</td></tr>"
            rows.append(row)
    rows_joined = '\n'.join(rows)
    table = f"<table><tbody><tr><td><b>Bewertungstext</b>&nbsp;</td><td><b>Vorhersage</b>&nbsp;</td><td><b>Tatsächlich</b>&nbsp;</td></tr>{rows_joined}</tbody></table>"
    display(HTML(table))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.YlGnBu):
    """Funktion zur Anzeige der Wahrheitsmatrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
