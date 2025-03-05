import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Charger le modèle sauvegardé et le scaler
tree_model = joblib.load('arbre_decision_model.joblib')

# Fonction pour effectuer une prédiction
def predict_default(age, income, home_ownership, emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, cb_person_default_on_file, cb_person_cred_hist_length):
    input_data = np.array([[
        age, income, home_ownership, emp_length, loan_intent, loan_grade,
        loan_amnt, loan_int_rate, cb_person_default_on_file, cb_person_cred_hist_length
    ]])

    # Standardiser les données d'entrée
    input_data_scaled = scaler.transform(input_data)

    # Prédiction
    prediction = tree_model.predict(input_data_scaled)
    return prediction[0]

# Interface utilisateur Streamlit
st.title("Prédiction du Risque de Crédit")
st.subheader("Saisissez les informations de l'emprunteur pour prédire le risque de défaut de paiement.")

# Widgets pour la saisie des données par l'utilisateur
age = st.slider("Âge de l'emprunteur", 20, 100, 30)
income = st.number_input("Revenu annuel de l'emprunteur (en $)", min_value=1000, max_value=1000000, value=50000)
home_ownership = st.selectbox("Type de logement", ("Locataire", "Propriétaire", "Autre"))
home_ownership = 0 if home_ownership == "Locataire" else 1 if home_ownership == "Propriétaire" else 2
emp_length = st.slider("Ancienneté de l'emploi (en années)", 0, 40, 5)
loan_intent = st.selectbox("But du prêt", (0, 1, 2, 3, 4, 5))
loan_grade = st.selectbox("Note du prêt", (0, 1, 2, 3, 4, 5))
loan_amnt = st.number_input("Montant du prêt (en $)", 500, 35000, 10000)
loan_int_rate = st.slider("Taux d'intérêt du prêt (%)", 5.0, 25.0, 10.0)
cb_person_default_on_file = st.selectbox("Historique de défaut de paiement", ("Non", "Oui"))
cb_person_default_on_file = 0 if cb_person_default_on_file == "Non" else 1
cb_person_cred_hist_length = st.slider("Longueur de l'historique de crédit (en années)", 2, 30, 5)

# Lorsque l'utilisateur clique sur le bouton pour faire une prédiction
if st.button("Prédire le statut du prêt"):
    # Effectuer la prédiction
    prediction = predict_default(age, income, home_ownership, emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, cb_person_default_on_file, cb_person_cred_hist_length)

    # Affichage du résultat
    if prediction == 1:
        st.error("Risque élevé de défaut de paiement")
    else:
        st.success("Prêt approuvé (aucun risque de défaut)")
