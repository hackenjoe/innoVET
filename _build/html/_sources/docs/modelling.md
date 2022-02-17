---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(modelling)=

# Modellierung
```{admonition} Erklärung zum Prozessschritt (Aufklappen)
:class: dropdown
Dieser Teil stellt den Kern der Datenanalyse dar. Eine datengetriebene KI-Lösung erfordert immer ein Lernmodell. Dieses Lernmodell gilt es mit ausreichend hoher Datenmengen zu trainieren. Durch das Training wird das Lernmodell besser in der Ausführung seiner Aufgabe. Entscheidend hier ist die Auswahl eines geeigneten Modells, dass am besten die Anforderungen aus der Problembeschreibung erfüllt. Wie das Schaubild aus S.1 zeigt, kann die Modellierung zur Folge haben, dass von hier aus ein Schritt zurück in die Datenaufbereitungsphase gesprungen werden muss (beispielsweise wenn die Daten nicht wie geplant für das Modell funktionieren oder das gewählte Lernmodell bestimmte Eigenschaften an den Daten nachträglich voraussetzt). Daher sind die zentralen Schritte in dieser Phase:

* Auswahl eines geeigneten Modells und dessen Implementierung
* Erstellung eines Testmodells zur Verifikation
* Modellbewertung im Hinblick auf die betriebswirtschaftliche Problemstellung

Die THU bietet verschiedene Modellentwicklungen für die unterschiedlichsten Problemstellungen an. Gerne setzen wir uns mit Ihnen zusammen, um gemeinsam ein geeignetes individuelles Modell für Ihre Problemstellung zu formulieren.
```

```{admonition} Handwerkerbewertungen:
:class: tip
Für die Handwerkerbewertungen eignet sich hier insbesondere ein Klassifizierungsmodell aus dem Bereich der natürlichen Sprachverarbeitung, da hier Freitexte von Kunden verwendet werden, die keine näheren Limitierungen aufweisen.
```



```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
from sentimental_imports import *
from sentimental_class import SentimentModel

# Auswahl des Modells
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# Einen Tokenizer für das Modell
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Eine Pipeline, die alles verbindet und die Bedienbarkeit vereinfacht
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0 if str(device) == 'cuda' else -1)
# Simpler Test
classifier("Ich finde, dass die Handwerksbetriebe gute Arbeit vollrichten")[0]['label']
```

Wie wir sehen können, wurde unser Testbeispiel mit einer Bewertung von $4$ Sternen bewertet. Ändern wir nun eine kleine Stelle in diesem Beispiel:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
classifier("Ich finde, dass die Handwerksbetriebe sehr gute Arbeit vollrichten")[0]['label']
```

Durch die Änderung von *gut* durch *sehr gut* hat sich das Rating um einen Stern verbessert. Dies ist intuitiv auch nachvollziehbar, da eine gute Bewertung eben noch keine sehr gute Bewertung darstellt. Das Modell scheint also auch für deutsche Texte zu funktionieren. Für unseren Anwendungsfall, wollen wir ja nur wissen, ob jemand etwas positives, neutrales oder negatives geschrieben hat. Also sollten wir noch ein paar kleinere Ergänzungen einfügen und diese am besten kompakt in einer Klasse integrieren:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
bewertungstexte = ["Miserabler Handwerker!", "aber insgesamt bin ich sehr zufrieden", "Das war unfair", "Das ist gar nicht mal so gut",
                   "Total awesome!","nicht so schlecht wie erwartet", "Das ist gar nicht mal so schlecht",
                   "Der Test verlief positiv.", "Der Verputzer hat die Arbeit ganz okay ausgeführt.", "Der Elektriker war sehr zuvorkommend. Gerne wieder!"]

model_b = SentimentModel(model_name)

for i,j in zip(bewertungstexte, model_b.predict_sentiment(bewertungstexte)):
    print(f'{i} --> {j}')
```

Das sieht schon sehr gut aus! Wir können offenbar kurze Texte bereits klassifizieren. Die Testbewertungen würden wir als Mensch ebenso interpretieren. Dennoch, um ein wirklich zuverlässiges Modell auch für längere oder komplizierterer Bewertungstexte zu valideren, benötigen wir noch mehr Daten. Glücklicherweise haben wir ja unsere aufbereiteten Handwerkerbewertungen. Guter Zeitpunkt, um diese Daten mit unserem neuen Modell zu testen.

```{admonition} Hinweis:
:class: note
Das geladene Modell ist ein starkes vortrainiertes Modell mit einem ausreichend großen Textkorpus (u.a. mehrsprachig und mit $137.000$ deutscher Produktbewertungen trainiert). Die *Handwerkerbewertungen* sind jedoch nicht Teil dieses Modells. Daher wird diese Art des Lernens auch als *transferes Lernen* bezeichnet. Denn wir wenden das Modell für Daten einer Klasse an, die es zuvor noch nie gesehen hat. 
```

```{exercise} Ihre Aufgaben
:class: dropdown
:nonumber: true
* Aus dem Code ist ersichtlich, dass ein *Tokenizer* für das Modell verwendet wird. Was ist darunter zu verstehen? Warum benötigen wir sowas und wie sehen die Texte nach der Tokenisierung für BERT aus?
* Welche Vorteile ergeben sich aus dem Einsatz von `cuda`? Prüfen Sie Ihre PyTorch Installation auf Unterstützung von CUDA.
* Laden Sie das vortrainierte Modell [**BERT-base-multilingual-uncased-sentiment**](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) in ein neues Python Skript und testen Sie eigene Bewertungstexte.
* Finden Sie Sätze die Ihrer Meinung nach einer anderen Gefühlslage entsprechen sollten? Wenn ja, woran könnte das Problem liegen?
```