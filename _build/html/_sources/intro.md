---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Data Lab: Gefühlsanalyse (Musterbeispiel)

> **Szenario:** Gute Bewertungen von Kunden sind ein entscheidender Erfolgsfaktor für die Akquise neuer Kunden. Denn niemand kauft gerne in Etablissements ein, bei denen es negative Rezensionen hagelt. Doch nicht immer äußern Kunden ihre Zufriedenheit auf Bewertungsportalen mit 1-5 Sterneskalen. Oftmals werden Beschwerdemails an den Support geschickt oder Beiträge in Foren bzw. über Messengers verschickt. Hier fehlt häufig eine Rankingfunktion, so dass nicht immer eindeutig ist, ob der Kunde zufrieden oder eher unzufrieden mit der erbrachten Dienstleistung ist. Umso wichtiger ist daher eine fundierte Analyse solcher Bewertungstexte. Eine Möglichkeit hierfür stellt die Gefühlsanalyse dar, die wir Ihnen nachfolgend am Beispiel von Handwerksbewertungen näher erklären.

Für die Struktur und als Leitfaden wenden wir das anerkannte Standardprozessvorgehensmodell [CRISP-DM](https://s2.smu.edu/~mhd/8331f03/crisp.pdf) an. CRISP-DM steht hierbei für *Cross Industry Standard Process Model for Data Mining* und ist ein iteratives Vorgehensmodell für KI-Projekte. (Seine Wurzeln kommen aus dem deutschen Raum, da u.a. die Daimler AG bei der Entwicklung mitwirkte [(Chapman et al., 1999)](https://the-modeling-agency.com/crisp-dm.pdf)). Das Modell besteht aus 6 individuellen Phasen die alle durchlaufen werden müssen:

```{figure} /_static/crisp_german.png
:scale: 45%
:name: crisp-dm

Prozessvorgehensmodell CRISP-DM
```

Eine detailierte Beschreibung von jeder Phase kann [hier](https://the-modeling-agency.com/crisp-dm.pdf) nachgelesen werden. Das vorliegende Dokument ist analog zu den Phasen aus dem Schaubild aufgebaut. Sie finden sie über das linke Navigationsmenü und können zwischen den einzelnen Schritten wechseln. Jede Phase wird dabei eingangs kurz beschrieben und erklärt.

```{admonition} Hinweis:
:class: note
Die verwendeten Testdaten für die Analyse stammen von [GoLocal](https://www.golocal.de/deutschland/reviews/handwerker/) und wurden nur aus Bildungsaspekten und Demonstrationszwecken eingesetzt. Die eigentlichen Trainingsdaten des Modells stammen *nicht* von dieser Seite.
```
