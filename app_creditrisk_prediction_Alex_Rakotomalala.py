import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le modèle sauvegardé
model = joblib.load('arbre_decision_model.joblib')
scaler = StandardScaler()

# Fonction pour effectuer la prédiction
def predict_client(default, age, income, loan_amount, int_rate, emp_length, home_ownership, loan_intent, loan_grade, credit_hist_length):
    # Prétraiter les données
    data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [int_rate],
        'person_emp_length': [emp_length],
        'person_home_ownership': [home_ownership],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'cb_person_cred_hist_length': [credit_hist_length]
    })
    
    # Appliquer la même normalisation
    data_normalized = scaler.transform(data)
    
    # Effectuer la prédiction
    prediction = model.predict(data_normalized)
    
    return prediction[0]

# Interface utilisateur
st.title("Prédiction du Risque de Crédit")

# Demander à l'utilisateur de fournir les caractéristiques du client
age = st.slider("Âge du client", 18, 100, 30)
income = st.number_input("Revenu annuel du client (en USD)", min_value=0, value=50000)
loan_amount = st.number_input("Montant du prêt (en USD)", min_value=0, value=10000)
int_rate = st.number_input("Taux d'intérêt du prêt (%)", min_value=0.0, value=5.0)
emp_length = st.slider("Ancienneté professionnelle (en années)", 0, 50, 5)

# Variables catégorielles (à transformer en numériques)
home_ownership = st.selectbox("Type de logement", ["Home", "Mortgage", "Rent"])
loan_intent = st.selectbox("Intention du prêt", ["Personal", "Family", "Educational", "Business"])
loan_grade = st.selectbox("Grade du prêt", ["A", "B", "C", "D", "E", "F", "G"])
credit_hist_length = st.slider("Historique de crédit (en années)", 0, 50, 10)

# Convertir les variables catégorielles en valeurs numériques
encoder = LabelEncoder()
home_ownership = encoder.fit_transform([home_ownership])[0]
loan_intent = encoder.fit_transform([loan_intent])[0]
loan_grade = encoder.fit_transform([loan_grade])[0]

# Bouton pour effectuer la prédiction
if st.button("Prédire le statut du prêt"):
    # Effectuer la prédiction
    result = predict_client(default=0,  # Le label 'default' n'est pas nécessaire ici, il est juste pour l'exemple.
                            age=age,
                            income=income,
                            loan_amount=loan_amount,
                            int_rate=int_rate,
                            emp_length=emp_length,
                            home_ownership=home_ownership,
                            loan_intent=loan_intent,
                            loan_grade=loan_grade,
                            credit_hist_length=credit_hist_length)
    
    # Afficher le résultat
    if result == 1:
        st.write("Le client présente un risque de défaut sur son prêt.")
    else:
        st.write("Le client ne présente pas de risque de défaut sur son prêt.")

