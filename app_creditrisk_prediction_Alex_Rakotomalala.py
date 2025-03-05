import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le modèle
model = joblib.load("arbre_decision_model.joblib")

# Charger et prétraiter les données
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Remplissage des valeurs manquantes
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

# Encodage des variables catégorielles
encoder = LabelEncoder()
df['person_home_ownership'] = encoder.fit_transform(df['person_home_ownership'])
df['loan_intent'] = encoder.fit_transform(df['loan_intent'])
df['loan_grade'] = encoder.fit_transform(df['loan_grade'])
df['cb_person_default_on_file'] = encoder.fit_transform(df['cb_person_default_on_file'])

# Normalisation
scaler = StandardScaler()
X = df.drop(columns=['loan_status'])
X_scaled = scaler.fit_transform(X)

# Interface utilisateur
st.title("Détection du Risque de Crédit")
st.write("Entrez les informations du client pour prédire s'il est risqué ou non.")

# Entrée utilisateur
person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
person_income = st.number_input("Revenu", min_value=1000, max_value=1000000, value=50000)
person_home_ownership = st.selectbox("Type de logement", df['person_home_ownership'].unique())
person_emp_length = st.number_input("Durée d'emploi (années)", min_value=0, max_value=50, value=5)
loan_intent = st.selectbox("Motif du prêt", df['loan_intent'].unique())
loan_grade = st.selectbox("Grade du prêt", df['loan_grade'].unique())
loan_amnt = st.number_input("Montant du prêt", min_value=1000, max_value=100000, value=10000)
loan_int_rate = st.number_input("Taux d'intérêt", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Pourcentage du revenu consacré au prêt", min_value=0.0, max_value=1.0, value=0.2)
cb_person_default_on_file = st.selectbox("Historique de défaut de paiement", df['cb_person_default_on_file'].unique())
cb_person_cred_hist_length = st.number_input("Longueur de l'historique de crédit", min_value=0, max_value=30, value=5)

# Encodage des entrées utilisateur
input_data = pd.DataFrame([[person_age, person_income, person_home_ownership, person_emp_length,
                            loan_intent, loan_grade, loan_amnt, loan_int_rate,
                            loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length]],
                          columns=X.columns)
input_data_scaled = scaler.transform(input_data)

# Prédiction
if st.button("Prédire le risque de crédit"):
    prediction = model.predict(input_data_scaled)
    st.write("Résultat :", "Client risqué" if prediction[0] == 1 else "Client non risqué")

# Visualisations
st.subheader("Visualisation des données")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['person_age'], bins=30, kde=True, color='blue', ax=axes[0])
axes[0].set_title("Distribution de l'âge")
sns.boxplot(x=df['loan_amnt'], color='orange', ax=axes[1])
axes[1].set_title("Montant du prêt")
st.pyplot(fig)
