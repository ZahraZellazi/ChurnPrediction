import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Chargement du modèle et des encodeurs
model = joblib.load("rf_model.pkl")
le_state = joblib.load("state_encoder.pkl")  # Charger l'encodeur pour 'state'

# Fonction pour convertir 'Yes'/'No' en 1/0
def convert_yes_no(value):
    if value.lower() == 'yes':
        return 1
    elif value.lower() == 'no':
        return 0
    else:
        raise ValueError("La valeur doit être 'Yes' ou 'No'")

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Vérifiez ce qui est envoyé dans le formulaire
        data = request.form
        print("Form Data:", data)

        if 'state' not in data:
            raise ValueError("La clé 'state' est manquante dans les données du formulaire.")
        
        state_value = data['state']
        print(f"State Value: {state_value}")  # Affiche la valeur du champ state
        
        # Transformer 'state' avec le LabelEncoder
        state_index = le_state.transform([state_value])[0]  # Utiliser l'encodeur pour 'state'
        
        features = [
            state_index,  # Remplacer par l'index de l'état
            convert_yes_no(data['international_plan']),
            float(data['area_code']),
            float(data['total_day_minutes']),
            float(data['total_day_charge']),
            float(data['total_intl_calls']),
            float(data['customer_service_calls']),
        ]

        # Convertir features en format NumPy array 2D
        features = np.array(features).reshape(1, -1)

        
        # Faire la prédiction
        prediction = model.predict(features)
        result = "Churn" if prediction[0] == 1 else "No Churn"

        return render_template('result.html', result=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
