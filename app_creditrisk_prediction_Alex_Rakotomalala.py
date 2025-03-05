import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Chargement du modèle sauvegardé
model = joblib.load('arbre_decision_model.joblib')

# Chargement du dataset pour l'encoding
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Encodage des variables catégorielles pour la prédiction
encoder = LabelEncoder()
df['person_home_ownership'] = encoder.fit_transform(df['person_home_ownership'])
df['loan_intent'] = encoder.fit_transform(df['loan_intent'])
df['loan_grade'] = encoder.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = encoder.fit_transform(df['cb_person_default_on_file'])

# Standardisation du modèle
scaler = StandardScaler()

# Création de l'interface Streamlit

# Titre de l'application
st.title("Prédiction du Risque de Crédit")

# Sous-titre avec une description
st.subheader("Description de l'Application")
st.write("""
Cette application permet d'évaluer le risque de défaut d'un client en fonction de ses caractéristiques financières et personnelles. 
Les utilisateurs peuvent entrer des informations telles que l'âge, le revenu, le montant du prêt demandé, ainsi que d'autres facteurs importants.
L'application utilise un modèle de machine learning basé sur un **arbre de décision** pour prédire si un client présente un risque élevé de défaut de paiement sur son prêt.

### Fonctionnalités :
- Entrez les informations du client pour obtenir une prédiction en temps réel.
- Visualisez des graphiques d'analyse des données pour mieux comprendre les relations entre les variables.
- Découvrez les performances du modèle avec des courbes ROC et des analyses détaillées.

### Comment utiliser :
1. Remplissez les champs avec les informations demandées.
2. Cliquez sur "Prédire" pour obtenir le résultat en temps réel.
""")

# Demande des informations au client
st.header("Entrez les caractéristiques du client")

# Formulaire de saisie
age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
income = st.number_input("Revenu annuel du client", min_value=1000, max_value=1000000, value=30000)
emp_length = st.number_input("Longueur de l'expérience professionnelle (en années)", min_value=0, max_value=40, value=5)
loan_amnt = st.number_input("Montant du prêt demandé", min_value=1000, max_value=500000, value=50000)
loan_int_rate = st.number_input("Taux d'intérêt du prêt", min_value=0.01, max_value=30.0, value=5.0)

# Choix des variables catégorielles
home_ownership = st.selectbox("Type de logement", ["OWN", "MORTGAGE", "RENT"])
loan_intent = st.selectbox("Intention du prêt", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"])
loan_grade = st.selectbox("Note de crédit", ["A", "B", "C", "D", "E", "F", "G"])
default_on_file = st.selectbox("Présence de défaut sur le fichier", ["Y", "N"])

# Préparation des caractéristiques pour la prédiction
data = {
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_length,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "person_home_ownership": home_ownership,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "cb_person_default_on_file": default_on_file
}

# Encodage des caractéristiques catégorielles
data['person_home_ownership'] = encoder.transform([data['person_home_ownership']])[0]
data['loan_intent'] = encoder.transform([data['loan_intent']])[0]
data['loan_grade'] = encoder.transform([data['loan_grade']])[0]
data['cb_person_default_on_file'] = encoder.transform([data['cb_person_default_on_file']])[0]

# Standardisation des données
features = np.array(list(data.values())).reshape(1, -1)
features_scaled = scaler.fit_transform(features)

# Prédiction avec le modèle chargé
prediction = model.predict(features_scaled)

# Affichage du résultat
if prediction == 1:
    st.write("Le client présente un **risque de défaut**.")
else:
    st.write("Le client présente **un faible risque de défaut**.")

# Ajout de visualisations

# Graphiques : Distribution des âges des clients
st.subheader("Visualisation des données de distribution")
plt.figure(figsize=(10, 5))
sns.histplot(df['person_age'], bins=30, kde=True, color='blue')
plt.title("Distribution de l'âge des clients")
st.pyplot(plt)

# Boxplot pour les montants de prêts
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['loan_amnt'], color='orange')
plt.title("Boxplot du montant des prêts")
st.pyplot(plt)

# Heatmap de la corrélation
st.subheader("Visualisation des corrélations")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap des Corrélations")
st.pyplot(plt)

# Répartition de l'âge en fonction du statut du prêt
st.subheader("Répartition de l'âge en fonction du statut du prêt")
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="person_age", hue="loan_status", bins=30, kde=True, palette="viridis")
plt.title("Répartition de l'âge en fonction du statut du prêt")
st.pyplot(plt)

# Boxplot des taux d'intérêt en fonction du statut du prêt
st.subheader("Taux d'intérêt en fonction du statut du prêt")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["loan_status"], y=df["loan_int_rate"], palette="Set2")
plt.title("Taux d'intérêt en fonction du statut du prêt")
st.pyplot(plt)

# Graphiques de la performance du modèle (Courbe ROC)

from sklearn.metrics import roc_curve, auc

# Prédiction des probabilités pour la courbe ROC
y_prob = model.predict_proba(features_scaled)[:, 1]

# Calcul des taux de faux positifs et vrais positifs pour la courbe ROC
fpr, tpr, _ = roc_curve([0], y_prob)  # '0' est la valeur réelle, ici nous simulons avec une seule prédiction

# Affichage de la courbe ROC
st.subheader("Courbe ROC du modèle")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Courbe ROC")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Courbe aléatoire
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC")
plt.legend()
st.pyplot(plt)
