import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le modèle sauvegardé
tree_model = joblib.load("arbre_decision_model.joblib")

# Charger le dataset pour l'analyse exploratoire
df = pd.read_csv("credit_risk_dataset.csv", sep=";")

# Encodage des variables catégorielles
encoder_dict = {}
for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoder_dict[col] = encoder

# Standardisation des données
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=['loan_status']))

# Interface utilisateur avec Streamlit
st.title("Prédiction du Risque de Crédit")
st.write("Entrez les informations du client pour prédire le risque de crédit.")

# Création des champs pour entrer les données utilisateur
person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
person_income = st.number_input("Revenu annuel", min_value=1000, max_value=500000, value=50000)
person_home_ownership = st.selectbox("Type de logement", encoder_dict['person_home_ownership'].classes_)
person_emp_length = st.number_input("Années d'emploi", min_value=0.0, max_value=50.0, value=5.0)
loan_intent = st.selectbox("Intention du prêt", encoder_dict['loan_intent'].classes_)
loan_grade = st.selectbox("Grade du prêt", encoder_dict['loan_grade'].classes_)
loan_amnt = st.number_input("Montant du prêt", min_value=500, max_value=50000, value=10000)
loan_int_rate = st.number_input("Taux d'intérêt", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = loan_amnt / person_income
cb_person_default_on_file = st.selectbox("Antécédent de défaut de paiement", encoder_dict['cb_person_default_on_file'].classes_)
cb_person_cred_hist_length = st.number_input("Longueur de l'historique de crédit", min_value=0, max_value=30, value=5)

# Encodage des entrées utilisateur
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [encoder_dict['person_home_ownership'].transform([person_home_ownership])[0]],
    'person_emp_length': [person_emp_length],
    'loan_intent': [encoder_dict['loan_intent'].transform([loan_intent])[0]],
    'loan_grade': [encoder_dict['loan_grade'].transform([loan_grade])[0]],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [encoder_dict['cb_person_default_on_file'].transform([cb_person_default_on_file])[0]],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

# Standardiser les entrées utilisateur
input_data_scaled = scaler.transform(input_data)

# Bouton de prédiction
if st.button("Prédire le Risque de Crédit"):
    prediction = tree_model.predict(input_data_scaled)[0]
    prediction_proba = tree_model.predict_proba(input_data_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ Risque élevé de défaut de paiement (Probabilité: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Faible risque de défaut de paiement (Probabilité: {prediction_proba:.2f})")

# Visualisation des distributions
st.subheader("Visualisation des Données")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['person_age'], bins=30, kde=True, color='blue', ax=ax)
ax.set_title("Distribution de l'âge des clients")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['loan_amnt'], color='orange', ax=ax)
ax.set_title("Boxplot des montants de prêt")
st.pyplot(fig)

# Affichage de la matrice de corrélation
st.subheader("Matrice de Corrélation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)
