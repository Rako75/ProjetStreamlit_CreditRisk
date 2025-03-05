import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le modèle et le scaler
model = joblib.load("arbre_decision_model.joblib")
scaler = joblib.load("scaler.joblib")  # Charger le scaler utilisé pendant l'entraînement

# Charger le dataset
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Encoder les variables catégorielles (appliqué aussi aux nouvelles données)
encoder = LabelEncoder()
df['person_home_ownership'] = encoder.fit_transform(df['person_home_ownership'])
df['loan_intent'] = encoder.fit_transform(df['loan_intent'])
df['loan_grade'] = encoder.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = encoder.fit_transform(df['cb_person_default_on_file'])

# Séparer les features et la cible
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Créer une interface pour l'utilisateur
st.title("Prédiction du Risque de Crédit")
st.write("Entrez les informations suivantes pour prédire si un prêt sera remboursé ou non.")

# Interface utilisateur pour entrer les informations
person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
person_income = st.number_input("Revenu mensuel", min_value=0, value=5000)
loan_int_rate = st.number_input("Taux d'intérêt du prêt", min_value=0.0, value=10.0)
person_home_ownership = st.selectbox("Type de logement", ['0', '1', '2', '3'])  # Valeurs encodées
loan_intent = st.selectbox("Objectif du prêt", ['0', '1', '2'])  # Valeurs encodées
loan_grade = st.selectbox("Note du prêt", ['0', '1', '2', '3', '4', '5'])  # Valeurs encodées
cb_person_default_on_file = st.selectbox("Historique de défaut", ['0', '1'])  # Valeurs encodées

# Formater les données pour le modèle
user_data = np.array([person_age, person_income, loan_int_rate, person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file])

# Appliquer les mêmes transformations (encodage et mise à l'échelle)
user_data = user_data.reshape(1, -1)

# Encoder les variables catégorielles
user_data[0, 3] = encoder.transform([user_data[0, 3]])  # person_home_ownership
user_data[0, 4] = encoder.transform([user_data[0, 4]])  # loan_intent
user_data[0, 5] = encoder.transform([user_data[0, 5]])  # loan_grade
user_data[0, 6] = encoder.transform([user_data[0, 6]])  # cb_person_default_on_file

# Standardisation des nouvelles données en utilisant le même scaler
user_data_scaled = scaler.transform(user_data)

# Prédiction
prediction = model.predict(user_data_scaled)

# Afficher la prédiction
if prediction == 1:
    st.write("Le client présente un risque de défaut.")
else:
    st.write("Le client ne présente pas de risque de défaut.")
