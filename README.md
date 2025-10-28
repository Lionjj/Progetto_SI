# Classificazione malattie cardiache (UCI Heart Disease - subset Cleveland)

Script usato per addestrare modelli in grado di predirre malattie cardiache.
Automatizza il processo di addestramento, valutazione e preprocessing riducedno tempi di analisi e confronto tra modelli.

---

## Indice

- [Descrizione del progetto](#descrizione-del-progetto)
- [Come installare ed eseguire il progetto](#come-installare-ed-eseguire-il-progetto)
- [Come usare il progetto](#come-usare-il-progetto)
- [Risultati e metriche](#risultati-e-metriche)
- [Ringraziamenti e riferimenti](#ringraziamenti-e-riferimenti)
- [Autore e licenza](#autore-e-licenza)

---

## Descrizione del progetto

Per addestrare i modelli e individuare il migliore tra Random Forest, K-Nearest Neighbors (k-NN) e Decision Tree è stato scelto il Dataset UCI Heart Disease - subset Cleveland scaricabile dal seguente link:
https://archive.ics.uci.edu/dataset/45/heart+disease

Il dataset offre le schede clinche di pazienti soggetti a malattie cardiache. Ogni scheda presenta un totale di 76 atribuiti, tutti gli esperimenti gli esperimenti però ne sfruttano solo 14 (compresa questa pipeline). Maggiori informazioni sono presenti nel sopracitato link.

- **Obiettivo:** Usare gli attributi delle schede clinche dei pazienti (14 attribuiti del dataset Cleveland), per individuare il modello più adatto per classificare la **presenza (1) vs assenza (0)** di malattie cardiache (binarizzazione eseguita sull'attributo num);
- **Stack/linguaggi:** Python, scikit-learn, pandas, numpy, matplotlib, joblib;
- **Caratteristiche principali:**
  - **Pipeline**: imputazione (median/most_frequent) → scaling (num) / One-Hot (cat) → modello;
  - **Confronto modelli di classificazione**: KNN, Decision Tree, Random Forest (tuning con GridSearchCV, CV 5-fold stratificata);
  - **Valutazione**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix; salvataggio grafici e metriche;
  - **Riproducibile**: split hold-out 80/20, random_state fissati, artefatti salvati in out/.

> ![ROC - Random Forest](out/figures/roc_rf.png) > ![Precision–Recall - Random Forest](out/figures/pr_rf.png) > ![Confusion Matrix - Random Forest](out/figures/cm_rf.png)

---

## Come installare ed eseguire il progetto

### Requisiti

- Python >= 3.9
- pip (o uv/poetry, se preferisci)

### Setup rapido (myenv)

```bash
# Clona il repo
git clone https://github.com/Lionjj/Progetto_SI.git
cd Progetto_SI

# Crea e attiva un ambiente virtuale
python3 -m venv myenv
source myenv/bin/activate   # su Windows: myenv\Scripts\activate
deactivate # per uscire

# Aggiorna pip e installa dipendenze
python -m pip install -U pip
pip install -r requirements.txt   # oppure: pip install numpy pandas scikit-learn matplotlib joblib
```

## Come usare il progetto

### Struttura pricipale del progetto

```bash
.
├── data/raw/                      # contiene il dataset
├── src/
│   ├── models/read.py             # legge il dataset
│   ├── utils/preprocessing.py     # pipeline
│   └── train.py                   # training
├── main.py                        # entrypoint
├── checks.py                      # check coerenza
└── out/{models,figures,results}/  # artefatti generati
```

### Comadi principali

Confronto completo (KNN, Decision Tree, Random Forest) con Feature Engineering attivo:

```bash
python main.py knn tree rf
```

Solo Random Forest (modello di default):

```bash
python main.py
```

Senza Feature Engineering:

```bash
python main.py knn tree rf --no-fe
```

Check di coerenza con il modello migliore:

```bash
python checks.py
```

### Output

I **modelli** generati vengono memorizzati in:
`out/models/<nome_modello>.joblib`

Le **metriche e i risultati** in `out/results/metrics*<nome_modello>.json, cv*<nome_modello>.csv, summary.cvs`

I **grafici**: `out/figures/roc*<nome_modello>.png, pr*<nome*modello>.pnf, cm*<nome_modello>.png`.

## Risultati e metriche

Come metrica pricpale ci si è basati sulla **media armonica** (F1 score) per avere un equilibrio tra **precisione e richiamo** (precision e recall).
A seguire abbiamo **l'accurateza, la precisione, il recall, il ROC_AUC e PR-AUC**.

| Model | CV best F1 | Test F1 @0.5 | Test Acc @0.5 | ROC-AUC | Test F1 @best |
| ----: | ---------: | -----------: | ------------: | ------: | ------------: |
|    RF |     0.7923 |       0.9123 |        0.9180 |  0.9600 |        0.9286 |
|   KNN |     0.7995 |       0.9091 |        0.9180 |  0.9453 |        0.9091 |
|  Tree |     0.7418 |       0.7719 |        0.7869 |  0.8295 |        0.8148 |

_Nota_: I valori esatti dipendono dalla esecuzione del codice, quelli nella tabella sovrastante sono frutto di una run del codice. Ogni run potrebbe portare a valori leggermente diversi.

## Ringraziamenti e riferimenti

- **Dataset**: UCI Machine Learning Repository – Heart Disease (subset Cleveland);
- **Librerie**: scikit-learn, pandas, numpy, matplotlib, joblib.

## Autore e licenza

Il seguente porgetto è stato realizzato da Ivone Danilo (Università degli studi di Bari: Aldo Moro).

[LICENSE](LICENSE)
