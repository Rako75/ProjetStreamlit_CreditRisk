import streamlit as st
import joblib
import numpy as np

# Charger les encodeurs et les modèles
encoder_home_ownership = joblib.load('encoder_home_ownership.pkl')
encoder_loan_intent = joblib.load('encoder_loan_intent.pkl')
encoder_loan_grade = joblib.load('encoder_loan_grade.pkl')
encoder_default_on_file = joblib.load('encoder_default_on_file.pkl')

log_reg = joblib.load('log_reg_model.pkl')
tree = joblib.load('tree_model.pkl')

# Fonction de transformation des entrées de l'utilisateur
def encode_user_input(home_ownership, loan_intent, loan_grade, default_on_file):
    # Encodage de l'input utilisateur
    try:
        home_ownership_encoded = encoder_home_ownership.transform([home_ownership])[0]
    except ValueError:
        home_ownership_encoded = -1  # Valeur par défaut si label inconnu

    try:
        loan_intent_encoded = encoder_loan_intent.transform([loan_intent])[0]
    except ValueError:
        loan_intent_encoded = -1

    try:
        loan_grade_encoded = encoder_loan_grade.transform([loan_grade])[0]
    except ValueError:
        loan_grade_encoded = -1

    try:
        default_on_file_encoded = encoder_default_on_file.transform([default_on_file])[0]
    except ValueError:
        default_on_file_encoded = -1

    return np.array([home_ownership_encoded, loan_intent_encoded, loan_grade_encoded, default_on_file_encoded])

# Interface Streamlit
st.title("Prédiction du Risque de Crédit")
st.subheader("Cette application permet de prédire si un client présente un risque de défaut de paiement en fonction de ses caractéristiques.")

# Formulaire de saisie des caractéristiques du client
home_ownership = st.selectbox("Type de logement", ["OWN", "MORTGAGE", "RENT"])
loan_intent = st.selectbox("Intention du prêt", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "MEDICAL", "VENTURE"])
loan_grade = st.selectbox("Note de crédit", ["A", "B", "C", "D", "E", "F", "G"])
default_on_file = st.selectbox("Présence de défaut de paiement", ["Y", "N"])

# Encodage des données de l'utilisateur
user_input = encode_user_input(home_ownership, loan_intent, loan_grade, default_on_file)

# Standardisation de l'entrée utilisateur
scaler = joblib.load('scaler.pkl')  # Si vous avez sauvegardé le scaler également
user_input_scaled = scaler.transform([user_input])

# Prédiction avec le modèle choisi (Logistique ou Arbre de Décision)
model_choice = st.radio("Choisir un modèle pour la prédiction", ("Régression Logistique", "Arbre de Décision"))
if model_choice == "Régression Logistique":
    prediction = log_reg.predict(user_input_scaled)
else:
    prediction = tree.predict(user_input_scaled)

# Afficher le résultat de la prédiction
if prediction == 0:
    st.write("Le client n'est pas à risque de défaut de paiement.")
else:
    st.write("Le client est à risque de défaut de paiement.")

# Affichage des probabilités (si nécessaire)
probability = log_reg.predict_proba(user_input_scaled)[:, 1]  # Utiliser le modèle logistique pour obtenir des probabilités
st.write(f"Probabilité de défaut de paiement : {probability[0]:.2f}")
