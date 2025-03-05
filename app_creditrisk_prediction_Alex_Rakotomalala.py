import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Charger les encodeurs et le modèle de l'arbre de décision
encoder_home_ownership = joblib.load('encoder_home_ownership.pkl')
encoder_loan_intent = joblib.load('encoder_loan_intent.pkl')
encoder_loan_grade = joblib.load('encoder_loan_grade.pkl')
encoder_default_on_file = joblib.load('encoder_default_on_file.pkl')

tree = joblib.load('arbre_decision_model.joblib')

# Charger les données
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Appliquer l'encodage sur les colonnes catégorielles du DataFrame
df['person_home_ownership'] = encoder_home_ownership.transform(df['person_home_ownership'])
df['loan_intent'] = encoder_loan_intent.transform(df['loan_intent'])
df['loan_grade'] = encoder_loan_grade.transform(df['loan_grade'])
df['cb_person_default_on_file'] = encoder_default_on_file.transform(df['cb_person_default_on_file'])

# Fonction de transformation des entrées de l'utilisateur
def encode_user_input(home_ownership, loan_intent, loan_grade, cb_person_default_on_file):
    try:
        home_ownership_encoded = encoder_home_ownership.transform([home_ownership])[0]
    except ValueError:
        home_ownership_encoded = -1  # Valeur par défaut si la catégorie est inconnue

    try:
        loan_intent_encoded = encoder_loan_intent.transform([loan_intent])[0]
    except ValueError:
        loan_intent_encoded = -1  # Valeur par défaut si la catégorie est inconnue

    try:
        loan_grade_encoded = encoder_loan_grade.transform([loan_grade])[0]
    except ValueError:
        loan_grade_encoded = -1  # Valeur par défaut si la catégorie est inconnue

    try:
        default_on_file_encoded = encoder_default_on_file.transform([cb_person_default_on_file])[0]
    except ValueError:
        default_on_file_encoded = -1  # Valeur par défaut si la catégorie est inconnue

    return np.array([home_ownership_encoded, loan_intent_encoded, loan_grade_encoded, default_on_file_encoded])

# Interface Streamlit
st.title("Prédiction du Risque de Crédit")
st.subheader("Cette application permet de prédire si un client présente un risque de défaut de paiement en fonction de ses caractéristiques.")

# Afficher des graphiques
st.subheader("Exploration des distributions (histogrammes)")
plt.figure(figsize=(15, 6))
for i, col in enumerate(["person_age", "person_income", "cb_person_cred_hist_length"]):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], bins=30, kde=True, color='blue')
    plt.title(f"Distribution de {col}")
st.pyplot(plt)

# Détection des valeurs aberrantes (boxplots)
st.subheader("Détection des valeurs aberrantes (boxplots)")
plt.figure(figsize=(15, 6))
for i, col in enumerate(["person_income", "loan_amnt"]):
    plt.subplot(1, 2, i+1)
    sns.boxplot(x=df[col], color='orange')
    plt.title(f"Boxplot de {col}")
st.pyplot(plt)

# Matrice de corrélation
st.subheader("Matrice de Corrélation")
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

# Relation entre le taux d'intérêt et le statut du prêt
st.subheader("Relation entre le taux d'intérêt et le statut du prêt")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["loan_status"], y=df["loan_int_rate"], palette="Set2")
plt.title("Taux d'intérêt en fonction du statut du prêt")
st.pyplot(plt)

# Influence du type de logement sur le risque de crédit
st.subheader("Influence du type de logement sur le risque de crédit")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="person_home_ownership", hue="loan_status", palette="coolwarm")
plt.title("Type de logement et statut du prêt")
st.pyplot(plt)

# Formulaire de saisie des caractéristiques du client
home_ownership = st.selectbox("Type de logement", ["OWN", "MORTGAGE", "RENT"])
loan_intent = st.selectbox("Intention du prêt", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "MEDICAL", "VENTURE"])
loan_grade = st.selectbox("Note de crédit", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.selectbox("Présence de défaut de paiement", ["Y", "N"])

# Encodage des données de l'utilisateur
user_input = encode_user_input(home_ownership, loan_intent, loan_grade, cb_person_default_on_file)

# Standardisation de l'entrée utilisateur
scaler = joblib.load('scaler.pkl')  # Charger le scaler
user_input_scaled = scaler.transform([user_input])

# Prédiction avec l'arbre de décision
prediction = tree.predict(user_input_scaled)

# Afficher le résultat de la prédiction
if prediction == 0:
    st.write("Le client n'est pas à risque de défaut de paiement.")
else:
    st.write("Le client est à risque de défaut de paiement.")

# Affichage des probabilités (si nécessaire)
probability = tree.predict_proba(user_input_scaled)[:, 1]  # Probabilité de défaut de paiement (classe 1)
st.write(f"Probabilité de défaut de paiement : {probability[0]:.2f}")
