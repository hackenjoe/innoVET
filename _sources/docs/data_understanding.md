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

(data_understanding)=

# Datenverständnis

```{admonition} Erklärung zum Prozessschritt (Aufklappen)
:class: dropdown


In diesem ersten datengetriebenen Teil des Data Science Projekts geht es um die Auswahl der relevanten Datensätze die analysiert werden sollen. Ein zentraler Bestandteil beim Verständnis der Daten ist die Beschaffung jener. Dies stellt häufig ein unterschätztes Problem dar. Beispielsweise müssen Rohdaten erst einmal aus den Zielquellen extrahiert und an einem Ort gebündelt, ggf. formatiert und geladen werden. Je nach Komplexität der Daten kann dieser Schritt ziemlich aufwendig werden. Als zweiter Baustein sollten die gesammelten und aggregierten Daten näher untersucht werden. Hierbei können wir sämtliche Werkzeuge aus der Statistik verwenden, um beispielsweise die Verteilung einzelner Begriffe oder Werte in den Daten festzustellen. Ziel ist es also ein allgemeines Verständnis - neben dem betriebswirtschaftlichen Verständnis aus der ersten Phase - über die Daten zu erlangen und zwar so, dass mit einem Experten aus dem Fachbereich darüber diskutiert werden kann. Zusammenfassend sind folgende Schritte wichtig:

* Datenallokation
* Datenauswertung und -beschreibung
* Datenbewertung hinsichtlich der Datenqualität

Wir vom KI-Kompetenzzentrum der THU verfolgen ebenfalls dieses Schema und analysieren Ihre Daten auf Anomalien, Fehler und Vollständigkeit. Es versteht sich von selbst, dass wir unsere Analysen durch anschauliche Auswertungen und Diagramme untermauern.

```

```{admonition} Handwerkerbewertungen:
:class: tip
Im vorliegenden Anwendungsfall können wir beispielsweise eruieren, ob unsere Daten fehlerhafte Werte enthalten. Ferner können wir uns Korrelationen, Duplikate, Werteverteilungen und Worthäufigkeiten anschauen.
```



#### Daten sammeln


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
import os
cwd = os.getcwd()
print(cwd)
from sentimental_imports import *

FILE_PATH = 'data/handwerksbewertungen.csv'
if not Path(FILE_PATH).is_file():
    gdd.download_file_from_google_drive(file_id='1k8E_OGxLptXSSA6rA1MhdbTjBrZhqa0J', dest_path=FILE_PATH)
df = pd.read_csv(FILE_PATH)
```


```{admonition} Hinweis:
:class: note
Oftmals sind die auszuwertenden Daten nicht lokal verfügbar, sondern müssen von zentralen Sammelstellen (entfernte Server, Cloud o.Ä.) bezogen werden und manchmal müssen dafür erst Zugangsrechte erlangt werden. 
```


#### Explorative Analyse
Nachdem wir die Handwerkerbewertungen in unser Skript geladen haben, wird es Zeit diese näher zu untersuchen. Da wir nicht wissen was uns erwartet, betrachten wir zunächst einen Ausschnitt dieser Daten.

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
display(df)
```

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
print(set(list(df.Ranking)))
```

Wir haben also 30.662 Handwerkerbewertungen. Dies stellt eine solide Basis für die spätere Modellentwicklung dar. Interessanterweise sind die Daten nicht durchgehend konsistent. So gibt es wohl vereinzelnte Bewertungen mit einer Bewertung außerhalb der 1-5 Skala. Hier kann davon ausgegangen werden, dass es sich um fehlerhafte Einträge handelt. Werfen wir mal, ganz explorativ, einen Blick auf einen solchen Datensatz:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
# Das sieht nicht nach normalen Bewertungen aus...
df_mask = df[~df.Ranking.isin([1,2,3,4,5])]
df_mask
```

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
x = df.shape[0]
df = df[df.Ranking.isin([1,2,3,4,5])]
df.reset_index(drop=True, inplace=True)
print(f"Entfernte Datensätze aufgrund Fehler: {x - df.shape[0]}")
```

Als nächstes überprüfen wir die Texte auf fehlende Werte und Duplikate. Falls welche gefunden werden, können wir diese ebenfalls von unserer Datenmenge entfernen, da sie für uns keine Relevanz haben:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
df = df[~df.Bewertungstext.isna()]
x = df.shape[0]
df.drop_duplicates(subset=["Bewertungstext"], inplace=True)
print(f"{x - df.shape[0]} doppelte Einträge gefunden")
```

Wir können uns auch in einer grafischen Übersicht die Datenverteilungen sowie weitere Metriken betrachten:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_notebook_iframe()
```
Hier interessieren uns vor allem die Variablen und die Datentypen. Wir sehen das neben des Rankings und des Bewertungstextes es auch eine Indexvariable gibt. Diese fungiert quasi als Zeiger für unsere Daten. Ferner wissen wir, dass der Bewertungstext und das Ranking kategoriale Variablen sind. Bewertungstext hat hierbei eine weitere Besonderheit: es gibt keine doppelten sowie keine fehlenden Texte. Dies ist auch nicht verwunderlich, da wir diese zuvor aus unseren Daten entfernt hatten. Interessant ist auch die Verteilung des Rankings. Offensichtlich gab es auf GoLocal deutlich mehr 5 Sterne- und 1 Sternebewertungen als Bewertungen mit 2, 3 oder 4 Sternen. Hier könnte man natürlich vermuten, dass nicht alle 5 Sternebewertungen auch wirklich von Kunden stammen. Wie auch immer, diese Übersichten und Visualisierungen geben uns einen relativ detailierten Einblick in unseren Daten und können als Indikator zur Feststellung der Datenqualität genutzt werden.

```{admonition} Hinweis:
:class: note
In dem oben gezeigten Pandas Profiling Bericht lassen sich auch Histogramme und Tortendiagramme anzeigen. Ein Histogramm ist eine grafische Darstellung der Häufigkeitsverteilung kardinal skalierter Merkmale. 
```
Betrachten wir aber zunächst eine gute, neutrale und schlechte Rezension, damit wir ein Gefühl dafür entwickeln, wie die Bewertungen aussehen:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
# Eine gute Bewertung
df[df.Ranking == 5].head(1)
```

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
# Eine neutrale Bewertung
df[df.Ranking == 3].head(1)
```

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
# Eine negative Bewertung
df[df.Ranking == 1].head(1)
```

Wir können uns auch die längste sowie kürzeste Bewertung anzeigen lassen:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
df['Anzahl Wörter'] = df.apply(lambda x: len(x.Bewertungstext.split()), axis=1)
display("Kürzeste Bewertung", df.sort_values(by="Anzahl Wörter", ascending=False).tail(1))
display("Längste Bewertung", df.sort_values(by="Anzahl Wörter", ascending=False).head(1))
print(f'Durchschnittliche Textlänge {np.sum(df["Anzahl Wörter"])/df.shape[0]:.0f} Wörter')
```

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
x = pd.DataFrame(df.groupby(["Anzahl Wörter"])['Anzahl Wörter'].count()).rename(columns={"Anzahl Wörter":"Wortlänge"}).reset_index()
x.columns = ["Anzahl Wörter im Satz", "Häufigkeit"]
axes = x.head(40).plot.bar(x="Anzahl Wörter im Satz", y="Häufigkeit", color="grey", figsize=(10, 7), title="Häufigkeit der Textlänge")
axes.set_ylabel("Anzahl Bewertungen")
axes.set_xlabel("Bewertungstext mit x Wörtern")
axes.set_yscale("log")
axes.legend(loc=1)
plt.show()
```

Wie aus dem Schaubild oben erkennbar ist, besteht die häufigste Bewertung aus gerade mal 8 Wörtern. Dies sind meist kurze und bündige Bewertungen wie z.B. "Toller Handwerksbetrieb und nette Mitarbeiter". Interessant für uns sind aber die Ausreißer, die es vor allem bei unter 5 Wörtern gibt. Eine Bewertung mit beispielsweise nur einem Wort ist eher fraglich und könnte auch auf einen Datenfehler hindeuten. Daher lassen wir uns einfach jene Bewertungen mit 4 oder wenigern Wörtern anzeigen:

```{code-cell} ipython3
---
{
    "tags": [
        "hide-input",
    ]
}
---
df.to_csv("bewertungen_roh.csv", index=False)
df[df["Anzahl Wörter"].isin([1,2,3,4])]
```

Wir sehen also, dass die meisten Bewertungen mit weniger als 4 Wörter überhaupt keinen Sinn ergeben und daher auch von unserer Datenmenge entfernt werden sollten.

```{exercise} Ihre Aufgaben
:class: dropdown
:nonumber: true
* Im Hinblick auf die Datenqualität scheinen kurze Texte oder sehr lange Texte fehlerbehaftet zu sein. Welche Ursachen könnten dazu geführt haben?
* In dem Pandas Profiling Bericht sehen wir $5$ _warnings_. Was haben die zu bedeuten und wie beeinflussen sie die Datenqualität?
* Erweitern Sie das Jupyter notebook `DataLab0x_Gefühlsanalyse_DataUnderstanding.ipynb` um weitere Analysen zur Datenqualität. Kann ein Zusammenhang zwischen der Textlänge und dem Ranking hergestellt werden? 

**Hinweis:** Sie könnten z.B. das Häufigkeitsdiagramm von oben wieder verwenden.
```