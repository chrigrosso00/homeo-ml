# Guida Utente Operativa: Self-Healing MLOps Runtime

Questa guida fornisce le istruzioni passo-passo per installare, eseguire e testare l'intero runtime di **Self-Healing MLOps** sul tuo computer, sia in modalità locale che tramite i container Docker.

---

## 📋 Indice
1. [Prerequisiti di Sistema](#1-prerequisiti-di-sistema)
2. [Installazione dell'Ambiente Locale](#2-installazione-dellambiente-locale)
3. [Avvio dell'Infrastruttura con Docker Compose](#3-avvio-dellinfrastruttura-con-docker-compose)
4. [Esecuzione delle Demo Interattive a Ciclo Chiuso](#4-esecuzione-delle-demo-interattive-a-ciclo-chiuso)
5. [Esecuzione Manuale dei Singoli Componenti](#5-esecuzione-manuale-dei-singoli-componenti)
6. [Consultazione degli Artefatti e delle Dashboard](#6-consultazione-degli-artefatti-e-delle-dashboard)
7. [Test Automatici e Qualità del Codice](#7-test-automatici-e-qualità-del-codice)
8. [Risoluzione Problemi Frequenti (Troubleshooting)](#8-risoluzione-problemi-frequenti-troubleshooting)

---

## 1. Prerequisiti di Sistema

Prima di iniziare, assicurati di avere installato:
* **Linux** (consigliato Ubuntu 20.04+) o **macOS**.
* **Python 3.11** (gestito preferibilmente tramite [Miniforge / Conda](https://github.com/conda-forge/miniforge)).
* **Docker** (versione 24.0+) e **Docker Compose** (v2.0+).
* Circa **2 GB** di spazio libero su disco (per le immagini Docker e i parquet del dataset NYC Taxi).

---

## 2. Installazione dell'Ambiente Locale

### 2.1 Clone del Repository
```bash
git clone https://github.com/chrigrosso00/homeo-ml.git
cd homeo-ml
```

### 2.2 Creazione e Attivazione dell'Ambiente Conda
```bash
# Crea un ambiente Conda isolato con Python 3.11
conda create -n mlops-runtime python=3.11 -y

# Attiva l'ambiente
conda activate mlops-runtime
```

### 2.3 Installazione delle Dipendenze in Modalità Modificabile
```bash
# Installa il runtime e tutti gli strumenti di sviluppo (pytest, ruff, black)
pip install -e ".[dev]"

# (Opzionale) Genera artefatti baseline minimi se vuoi avviare subito Docker senza dataset reali
python -m scripts.generate_test_fixtures
```

---

## 3. Avvio dell'Infrastruttura con Docker Compose

Il sistema include uno stack completo pronto all'uso con broker Kafka-compatibile, gateway Traefik e istanze di serving.

### 3.1 Avvio di Tutti i Container
```bash
docker compose up -d
```

### 3.2 Verifica dello Stato dei Container
Esegui:
```bash
docker compose ps
```
Dovresti vedere **5 servizi attivi**:
1. `mlops-redpanda`: Broker di streaming eventi ad altissima velocità (compatibile Kafka).
2. `mlops-redpanda-console`: Interfaccia web per esplorare topic e messaggi di telemetria.
3. `mlops-traefik`: Reverse proxy con bilanciamento pesato dinamico (WRR).
4. `mlops-serving-prod`: Microservizio FastAPI con modello di produzione `v1.0.0`.
5. `mlops-serving-canary`: Microservizio FastAPI pronto per i modelli candidati `v1.1.0`.

### 3.3 Dashboard Web Disponibili nel Browser

| Servizio | URL nel Browser | Descrizione |
| :--- | :--- | :--- |
| **Gateway Traefik (Ingress)** | [http://localhost:80](http://localhost:80) | Punto di ingresso unificato per le chiamate API (smistate tra Prod e Canary). |
| **Traefik Dashboard** | [http://localhost:8081](http://localhost:8081) | Visualizzazione dei router e dei pesi percentuali correnti (es. 100/0, 95/5). |
| **Redpanda Console** | [http://localhost:8080](http://localhost:8080) | Ispezione in tempo reale dei topic Kafka (`mobility.telemetry`, `fraud.telemetry`). |
| **FastAPI Swagger (Prod)** | [http://localhost:8001/docs](http://localhost:8001/docs) | Documentazione interattiva OpenAPI del servizio di Produzione. |
| **FastAPI Swagger (Canary)** | [http://localhost:8002/docs](http://localhost:8002/docs) | Documentazione interattiva OpenAPI del servizio Canary. |

---

## 4. Esecuzione delle Demo Interattive a Ciclo Chiuso

Il modo più rapido ed educativo per vedere il sistema in azione è utilizzare lo script orchestratore `./demo.sh`.

### 4.1 Dominio Principale: Urban Mobility (NYC Taxi COVID Shock)
Simula il traffico taxi pre-lockdown, inietta lo shock improvviso della quarantena COVID di marzo 2020 (strade deserte, velocità aumentate del 40%), attiva il drift Evidently, riaddestra un modello candidato adattivo e promuove il Canary al 100% su Traefik:

```bash
# Modalità Automatica con pause educative di 2 secondi (CONSIGLIATA):
./demo.sh --domain mobility --mode auto

# Modalità Guidata (premi [INVIO] per avanzare ad ogni fase):
./demo.sh --domain mobility --mode step

# Modalità Veloce (esecuzione istantanea per CI/CD, senza pause):
./demo.sh --domain mobility --mode fast
```

### 4.2 Dominio Secondario: E-Commerce Fraud Detection (Attacco Botnet)
Simula 300 transazioni nominali, inietta un attacco botnet coordinato (registrazioni istantanee ed emulatori browser), diagnostica la fetta anomala e promuove il canary:

```bash
./demo.sh --domain fraud --mode step
```

---

## 5. Esecuzione Manuale dei Singoli Componenti

Se desideri eseguire singolarmente le pipeline di training, serving o monitoraggio, ecco i comandi da utilizzare:

### 5.1 Addestramento Modello di Mobilità (LightGBM + ONNX)
Addestra il modello di baseline su Q1 2019 ed esporta il grafo ONNX:
```bash
python -m src.models.train_mobility
```
*Gli artefatti vengono salvati in `artifacts/mobility/`.*

### 5.2 Valutazione Temporale del Drift su 2020 ed Evidently Report
Analizza il crollo delle performance durante il lockdown e produce i report statistici:
```bash
python -m scripts.evaluate_mobility_drift
```

### 5.3 Avvio Locale del Server FastAPI (senza Docker)
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5.4 Test di Predizione Manuale con `curl`

**Richiesta per Urban Mobility (Congestione Stradale / SLA):**
```bash
curl -X POST "http://localhost:8000/predict/mobility" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2020-03-25T14:30:00",
    "pickup_location_id": 161,
    "dropoff_location_id": 236,
    "trip_distance": 2.5,
    "passenger_count": 1
  }'
```
*Risposta attesa:*
```json
{
  "is_congested": false,
  "congestion_probability": 0.0821,
  "model_version": "v1.0.0"
}
```

**Richiesta per Fraud Detection (Rischio Frode Carta/Account):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1001,
    "signup_time": "2026-10-29T10:00:00",
    "purchase_time": "2026-10-29T10:00:02",
    "purchase_value": 140.0,
    "device_id": "DEVICE_999",
    "source": "Ads",
    "browser": "Chrome",
    "sex": "M",
    "age": 32,
    "ip_address": 732758368.0
  }'
```

---

## 6. Consultazione degli Artefatti e delle Dashboard

Al termine di una simulazione o di un ciclo di self-healing, puoi ispezionare i risultati generati:

1. **Report Grafico Interattivo di Drift (Evidently HTML)**:  
   Apri il file nel tuo browser preferito:
   ```bash
   # Su Linux:
   xdg-open artifacts/mobility/latest_drift_report.html
   # Su macOS:
   open artifacts/mobility/latest_drift_report.html
   ```
   Visualizzerai le curve di densità prima e dopo lo shock, i test di Wasserstein e le feature driftate.

2. **Audit Trail Forense JSON**:  
   Visualizza la sequenza immutabile delle decisioni prese dalla Macchina a Stati:
   ```bash
   cat artifacts/supervisor/audit_trail.json | python -m json.tool
   ```

3. **Configurazione Routing Dinamico Traefik**:  
   Controlla i pesi correnti applicati in tempo reale (es. `100% / 0%` o `0% / 100%`):
   ```bash
   cat config/traefik_dynamic.yml
   ```

---

## 7. Test Automatici e Qualità del Codice

Tutto il codice del progetto è coperto da una suite di test rigorosa e rispetta gli standard Python 3.11+.

### 7.1 Esecuzione Test Suite (Pytest)
```bash
pytest -v
```
*Tutti i 35 test unitari e di integrazione devono risultare verdi (passed).*

### 7.2 Verifica Stile e Formattazione
```bash
# Controllo linting (regole Ruff, import e bugbear):
ruff check .

# Controllo formattazione (Black con lunghezza riga 100):
black --check .
```

---

## 8. Risoluzione Problemi Frequenti (Troubleshooting)

### La porta 80 o 8080 risulta già occupata
Se ricevi un errore del tipo `Bind for 0.0.0.0:80 failed: port is already allocated`:
- Verifica se hai un server web locale attivo (es. Apache o Nginx): `sudo systemctl stop nginx` oppure `sudo systemctl stop apache2`.
- In alternativa, puoi modificare la porta esposta in `docker-compose.yml` (es. `"8085:80"` per Traefik).

### Messaggi di telemetria Kafka Connection Refused nei test
- È assolutamente **normale** se esegui i test `pytest` senza aver avviato Docker: il client di telemetria è progettato per essere asincrono e *fail-safe* (se Kafka non è raggiungibile, non blocca l'inferenza del modello e logga un warning).

### Fermare e Pulire l'Ambiente Docker
Quando hai finito, puoi arrestare tutti i servizi con:
```bash
docker compose down
```
Se desideri eliminare anche i volumi persistenti:
```bash
docker compose down -v
```

