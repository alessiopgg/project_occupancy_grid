# Occupancy Grid 2D da dati LiDAR e ZED

Questo progetto realizza una pipeline offline per costruire una **occupancy grid 2D** a partire da dati reali acquisiti da un rover dotato di **LiDAR 2D** e camera **ZED**.

La pipeline legge bag ROS2, sincronizza le scansioni LiDAR con l'odometria della ZED, ricostruisce i raggi nel frame globale e aggiorna una mappa probabilistica tramite una rappresentazione in **log-odds**.

---

## Obiettivo del progetto

L'obiettivo è ottenere una mappa 2D dell'ambiente in cui ogni cella della griglia viene classificata come:

- **unknown**: cella non osservata o incerta;
- **free**: cella attraversata dai raggi LiDAR;
- **occupied**: cella corrispondente all'endpoint di un raggio valido.

Il progetto è stato sviluppato con particolare attenzione alla separazione tra preprocessing, costruzione della mappa e strumenti di debug/explainability.

---

## Pipeline

La pipeline è composta da tre fasi principali.

### 1. Preprocessing

Il preprocessing legge i dati grezzi dal bag ROS2 e prepara una struttura dati utilizzabile dal mapping.

Operazioni principali:

- lettura dei topic `/scan` e `/zed/zed_node/odom`;
- gestione dei timestamp `bag_time` e `header_time`;
- interpolazione della posa ZED al tempo dello scan;
- conversione del quaternione in yaw;
- applicazione della trasformazione approssimata ZED--LiDAR;
- conversione dei raggi LiDAR in punti locali e globali;
- filtraggio dei range non validi, dei valori nulli e delle misure oltre il range operativo.

### 2. Mapping

La fase di mapping costruisce la occupancy grid a partire dai dati preprocessati.

Operazioni principali:

- calcolo dei bound spaziali della mappa;
- discretizzazione dello spazio in celle;
- inizializzazione della griglia in log-odds;
- ray tracing tramite algoritmo di Bresenham;
- aggiornamento delle celle attraversate come `free`;
- aggiornamento della cella finale del raggio come `occupied`;
- conversione finale in probabilità e classificazione discreta.

### 3. Debug ed explainability

Sono stati aggiunti strumenti di analisi visuale per comprendere meglio la formazione della mappa.

In particolare, la viewer step-by-step permette di confrontare:

- la mappa cumulativa aggiornata fino allo scan corrente;
- la mappa prodotta dalla singola osservazione;
- la distribuzione dei raggi LiDAR dello scan corrente;
- i raggi validi, i raggi pari a zero e i raggi oltre il limite operativo.

---

## Demo viewer

La viewer consente di osservare l'evoluzione della mappa scan dopo scan.

Inserire qui una GIF o un breve video della viewer:

![Demo viewer](media/viewer_demo.gif)

Se si preferisce usare un video MP4 invece di una GIF:

[Guarda la demo della viewer](media/viewer_demo.mp4)

---

## Esempio di risultato

Esempio di mappa classificata prodotta dalla pipeline:

![Occupancy grid classificata](images/occupancy_classified.png)

Esempio di mappa probabilistica:

![Occupancy probability map](images/occupancy_probability.png)

---

## File principali

| File | Descrizione |
|---|---|
| `preprocess_bag.py` | Legge il bag ROS2, sincronizza scan e odometria, interpola la posa e ricostruisce i punti LiDAR. |
| `build_occupancy_grid.py` | Costruisce la occupancy grid in log-odds, applica Bresenham e genera mappe probabilistiche/classificate. |
| `run_test_mapping.py` | Script principale per lanciare un esperimento completo di mapping. |
| `zero_ray_explainability_viewer_v2.py` | Genera la viewer step-by-step per analizzare raggi validi, raggi a zero e mappe cumulative. |
| `occupancy_grid_Poggesi_AARI.pdf` | Report completo del progetto. |

---

## Esecuzione

Prima di eseguire gli script, aggiornare il percorso del bag ROS2 all'interno di `run_test_mapping.py`.

Esecuzione della pipeline principale:

```bash
python run_test_mapping.py
```

Esecuzione della viewer di explainability:

```bash
python zero_ray_explainability_viewer_v2.py "path/al/bag_ros2"
```

Esempio su Windows:

```bash
python zero_ray_explainability_viewer_v2.py "D:\\tesi\\acquisizioni\\nome_bag"
```

---

## Output generati

Ogni esperimento salva i risultati in una cartella dedicata, in modo da non sovrascrivere le run precedenti.

Output principali:

| Output | Descrizione |
|---|---|
| `occupancy_probability.png` | Mappa probabilistica finale. |
| `occupancy_classified.png` | Mappa classificata in unknown/free/occupied. |
| `log_odds.npy` | Griglia numerica in log-odds. |
| `probability_grid.npy` | Griglia di probabilità. |
| `classified_grid.npy` | Griglia classificata. |
| `occupancy_grid_metadata.json` | Parametri del run, bound della mappa e riepilogo numerico. |
| `viewer.html` | Viewer HTML per l'analisi step-by-step. |

---

## Struttura consigliata della cartella

```text
project/
├── README.md
├── occupancy_grid_Poggesi_AARI.pdf
├── preprocess_bag.py
├── build_occupancy_grid.py
├── run_test_mapping.py
├── zero_ray_explainability_viewer_v2.py
├── images/
│   ├── occupancy_probability.png
│   └── occupancy_classified.png
└── media/
    └── viewer_demo.gif
```

---

## Note sui dati

Nei dati reali sono state osservate alcune criticità che possono influenzare la qualità della mappa:

- presenza di numerosi range LiDAR pari a zero;
- misure oltre il range operativo scelto;
- calibrazione approssimata tra ZED e LiDAR;
- possibili discontinuità o traslazioni nella stima della posa;
- assenza, nella versione attuale, di deskew raggio per raggio dello scan LiDAR.

Queste limitazioni non indicano necessariamente un errore nell'algoritmo di occupancy grid, ma riflettono la complessità dell'utilizzo di dati reali acquisiti da sensori fisici.

---

## Report completo

Il report completo del progetto è disponibile nel file:

```text
occupancy_grid_Poggesi_AARI.pdf
```

Nel report sono descritti in dettaglio:

- il preprocessing dei dati;
- la costruzione della occupancy grid;
- le scelte implementative;
- le limitazioni dei dati acquisiti;
- il caso studio indoor;
- il confronto tra risoluzioni diverse;
- la viewer step-by-step;
- esempi aggiuntivi in ambiente esterno.

---

## Autore

**Alessio Poggesi**
