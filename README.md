# 🐍 KaggleChallenges

![](https://img.shields.io/github/repo-size/FlorianTeich/KaggleChallenges)

![](https://img.shields.io/github/actions/workflow/status/FlorianTeich/KaggleChallenges/main.yml?branch=main)

![Microservice Architecture](/assets/trailer.svg)

Amalgamation of various exploration techniques and ML methods for several Kaggle challenges.
These notebooks might contain code from other sites/people.

## 🚀 Build entire infrastructure in Docker-Compose

```
docker-compose build
docker-compose up
```

## 🐢 Install all dependencies & KCU locally

```
git clone https://github.com/FlorianTeich/KaggleChallenges.git
cd KaggleChallenges
pipenv install
pipenv shell
cd src
pip install -e .
```

## 🔥 Microservices

| Service | Port | Description |
| --- | --- | --- |
| jupyterlab (pyspark included) | 8889:8888 & 4040:4040 | Browser-based Python IDE |
| streamlit | 8501:8501 | Data Science app framework |
| postgres | 5432:5432 | SQL-Backend |
| neo4j | 7474:7474 & 7687:7687 | Graph-Database |
| minio | 9000:9000 & 9001:9001 | S3 Object Storage |
| redis | 6380:6380 | Key-Value-Store |
| mongo | 27017 & 27017 | NoSQL-Backend |
| mlflow | 5000:5000 | MLOps Service for storing & serving models as well as tracking model performance metrics |
| superset | 8060:8088 | Interactive visualizations & Dashboard creation tool |
| zookeeper |  | Maintenance tool |
| kafka | 9092:9092 & 29092:29092 | Data Streaming |
| kafka-connect | 8083:8083 & 5005:5005 | Change-Data-Capture for Postgres backend |
| loki | 3100:3100 | Logging |
| grafana | 3000:3000 | Visualiyation of Loki logs and metrics |

## 🦊 KCU - Kaggle-Challenges-Utilities Python Library

The KCU is a Python Library that offers boilerplate to communicate with all the microservices involved as well as streamlining certain processes (e.g. FeatureStore & ETL-Pipeline). 

## ✨ Current Kaggle Challenges

- [x] [Titanic](https://www.kaggle.com/c/titanic) ![](https://img.shields.io/badge/-tabular-blue)
- [x] [MNIST](https://www.kaggle.com/c/digit-recognizer) ![](https://img.shields.io/badge/-CV-blue)

## 🪣 Todo

- [ ] mypy
- [ ] coverage
- [ ] versioneer
- [ ] pre-commit-hook black
- [ ] isort
- [ ] kubernetes
