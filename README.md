**Contexte**
Nous sommes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. 

L’entreprise souhaite mettre en œuvre **un modèle de “scoring crédit” pour prédire la probabilité qu’un client rembourse son crédit**, puis classifie la demande en crédit accordé ou refusé. 

De plus, l'entreprise souhaite respecter une logique de transparence vis-à-vis des décisions d’octroi de crédit.

Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

En parallèle, l’entreprise veut s’inscrire dans une démarche MLOps moderne pour industrialiser le cycle de vie de son modèle, de l’entraînement initial jusqu’au déploiement et au monitoring en production.

**Objectifs**
***MISSION 1 :  Elaborer le modèle de scoring***

La mission 1 se déclinera autour des objectifs suivants :

`Construction d’un modèle de scoring crédit :`
* Prédire la probabilité de défaut de paiement d’un client.
* Classer automatiquement une demande de crédit en accordée ou refusée.
* Prendre en compte le déséquilibre des classes et le coût métier lié aux erreurs de prédiction (faux négatifs plus coûteux que les faux positifs).
* Définir un seuil de décision optimisé du point de vue métier.


`Interprétabilité du modèle :`
* Identifier les features les plus importantes globalement (feature importance globale).
* Fournir une explication locale des décisions pour chaque client (e.g., SHAP, LIME), afin d’assurer la transparence du modèle auprès des analystes.


`Industrialisation et déploiement du modèle (MLOps) :`
* Suivre les expérimentations et les performances via MLFlow.
* Enregistrer et servir les modèles avec le Model Registry de MLFlow.
* Mettre en œuvre une API de prédiction déployée sur une plateforme cloud via Github Actions et CI/CD.
* Intégrer des tests unitaires automatisés avec Pytest ou Unittest.
* Proposer une interface de test locale (via Streamlit ou notebook) pour simuler des requêtes vers l’API.


Michaël, notre manager, nous incite à sélectionner un ou des kernels Kaggle (des notebooks publics partagés par d'autres data scientists), pour nous faciliter l’analyse exploratoire, et nous aider à gagner du temps sur les premières étapes du projet, comme **L’analyse exploratoire des données (EDA)**, **La préparation des données (nettoyage, traitement des valeurs manquantes, encodage des variables, etc.)**, **Le feature engineering (création de nouvelles variables utiles pour le modèle).**

Nous nous sommes inspirés des kernels Kaggle suivants : 
* Pour l’analyse exploratoire : https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook
* Pour la préparation des données et le feature engineering : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script

***

***MISSION 2 :  Intégrez et optimisez le système MLOps***

Mickael revient vers nous deux semaines après notre première mission de scoring. Il souligne que le cycle de vie MLOps n’est pas encore complet, car une étape importante n’a pas encore été abordée : le suivi du modèle en production.

`Objectif principal :`
Tester la détection de dérive de données (Data Drift) avec la librairie evidently.
Cela permet de simuler une situation réelle en production, pour anticiper les écarts entre les données utilisées à l’entraînement et celles reçues une fois le modèle déployé.

`Tâche à réaliser :`
En prévision, il souhaiterait que nous testions l’utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production. Pour cela nous prendrons comme hypothèse que le dataset “application_train” représente les datas pour la modélisation et le dataset “application_test” représente les datas de nouveaux clients une fois le modèle en production. 
* application_train.csv : données utilisées pour entraîner le modèle.
* application_test.csv : données simulant les nouvelles données en production.

`L’analyse à l’aide d’evidently nous permettra de :`
* Comparer les deux jeux de données (les datas d’entraînement et les datas de production).
* Générer un rapport HTML qui permet de détecter les éventuels Data Drift sur les features principales.



**Ce projet a donc pour objectif de construire, suivre, déployer et monitorer un **modèle de scoring de crédit** permettant d’automatiser la prise de décision concernant l’octroi d’un prêt à un client.**

L’ensemble du projet inclut :
- Le **prétraitement des données** et la **modélisation Machine Learning**
- Le **tracking des expérimentations avec MLFlow**
- Le **déploiement du modèle sous forme d’API sur le cloud**
- L’**analyse du data drift** avec Evidently
- Une interface de **test client via Streamlit**

***

**Livrables**
Notebook de préparation des données et modélisation
API de scoring (code et documentation)
Dossier Dashboard contenant les fichiers liés au fonctionnement du dashboard
Interface Streamlit de test
Rapport Evidently de détection de dérive
Présentation finale du projet

***

**Structure du projet**
P7_Modele_Scoring/
│
├── data/                          # Données brutes et prétraitées
│   ├── data_train.csv
│   ├── application_train.csv
│   ├── application_test.csv
│
├── notebooks/                    # Notebooks de développement
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training_with_mlflow.ipynb
│
├── src/                          # Code source principal
│   ├── preprocessing.py
│   ├── merging.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── mlflow_tracking.py
│   ├── config.py
│   ├── utils.py
│   ├── Vizualisation.py
│   ├── data_drift_report.py
│
├── api/                          # Code de l’API FasApi
│   ├── app.py                    # Entrée principale de l’API
│   ├── model_utils.py           # Chargement - prédiction via le modèle
│
├── dashboards/                   # Outils d’analyse et interface de test
│   ├── data_drift_analysis.html # Rapport de drift Evidently
│   ├── streamlit_app.py         # Interface de test Streamlit
│
├── tests/                        # Tests unitaires
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│
├── .github/workflows/           # Intégration continue (CI/CD GitHub Actions)
│   ├── test_and_deploy.yml
│
├── mlruns/                       # Tracking des expérimentations MLflow (auto)
│
├── Procfile                      # Déploiement Heroku
├── requirements.txt             # Dépendances du projet
├── streamlit_app.py             # Point d’entrée
├── test_mlflow.py               # Test de tracking
├── .gitignore                   # Fichiers ignorés par Git
├── README.md                    # Documentation du projet

***

📬 Contact
Nom : Maty KANE

Email : matymbaye09@live.fr

GitHub : github.com/MatyKane

Projet réalisé dans le cadre du parcours Data Scientist – OpenClassrooms.