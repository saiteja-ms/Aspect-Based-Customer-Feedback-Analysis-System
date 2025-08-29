# NextPick — Real-time Recommendation Engine (DA5402 MLOps Project)

**Duration:** Mar 2025 – May 2025  
**Team Project:** DA5402  

NextPick is a **real-time recommendation system** built with collaborative filtering and a ranking model, deployed with full **MLOps** best practices. It provides **personalized recommendations**, **model versioning**, **drift detection**, and **monitoring** with a containerized, production-ready stack.

---

## ✨ Features
- **Recommendation Engine**
  - Candidate generation with **matrix factorization (collaborative filtering)**
  - Reranking with **LightGBM ranking model**
- **MLOps Pipeline**
  - **DVC** for dataset & model versioning
  - **MLflow** for experiment tracking and artifact logging
  - **Unit tests** with PyTest
  - **Automated CI** with GitHub Actions
- **Deployment**
  - **FastAPI** service for real-time recommendations
  - **PostgreSQL** backend for event logging and prediction storage
  - **Dockerized** service with `docker-compose`
- **Monitoring**
  - **Prometheus** for metrics collection
  - **Grafana** dashboards for visualization
  - API request counters, latencies, and custom business KPIs
- **Scalability**
  - Containerized → ready for **Kubernetes orchestration**
  - Modular design for pluggable models

---

## 📂 Project Structure
```
nextpick/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
├── dvc.yaml
├── params.yaml
├── sql/
│   └── init.sql
├── src/
│   ├── data/               # preprocessing
│   ├── models/             # collab + ranker
│   ├── train/              # training scripts
│   ├── evaluate/           # evaluation metrics
│   ├── serving/            # FastAPI app + metrics
│   ├── mlflow_tracking/    # MLflow helpers
│   └── tests/              # pytest unit tests
└── infra/
    ├── docker/
    └── k8s/
```

---

## ⚙️ Setup & Installation

### 1. Clone repo and install dependencies
```bash
git clone https://github.com/<your-username>/nextpick.git
cd nextpick
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialize DVC
```bash
dvc init
```

### 3. Start PostgreSQL + Prometheus + NextPick (docker-compose)
```bash
docker-compose up -d
```

---

## 🛠️ Data Pipeline (DVC)

### Preprocess data
```bash
dvc repro
```
Stages in `dvc.yaml`:
- **data_prep** → cleans raw events
- **train_collab** → trains collaborative filtering model
- **train_ranker** → trains LightGBM reranker
- **evaluate** → computes metrics (recall@k, ndcg)

---

## 🔬 MLflow Experiment Tracking
Launch MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
Logs:
- Model parameters
- Training metrics (AUC, NDCG, Recall@K)
- Saved artifacts (`.pkl` / `.joblib` models)

---

## 🚀 Serving API

Run locally:
```bash
uvicorn src.serving.api:APP --reload --host 0.0.0.0 --port 8080
```

Endpoints:
- `POST /recommend` → get recommendations for a user  
  ```json
  {
    "user_id": "u123",
    "top_k": 5
  }
  ```
  → returns ranked list of items
- `GET /metrics` → Prometheus metrics endpoint

---

## 📊 Monitoring

- Prometheus scrapes:
  - API request counts
  - Latency histograms
  - Custom gauges (e.g. candidate pool size)
- Grafana:
  - Dashboards for monitoring recommendation throughput, latency, drift

---

## 🧪 Testing

Run unit tests:
```bash
pytest -q
```

---

## 🐳 Deployment

### Docker (local)
```bash
docker build -t nextpick .
docker run -p 8080:8080 nextpick
```

### Docker Compose (with Postgres + Prometheus)
```bash
docker-compose up --build
```

### Kubernetes (optional)
Manifests in `infra/k8s/` can be applied:
```bash
kubectl apply -f infra/k8s/
```

---

## 📈 Drift Detection
- Scheduled jobs compare **train vs production distributions** using PSI / KS tests.
- Alerts pushed to **Slack** or monitoring system.

---

