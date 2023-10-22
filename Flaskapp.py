from flask import Flask, request, render_template
import joblib  # For loading your trained model
import numpy as np  # For data manipulation
import zipfile
#https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
app = Flask(__name__)

# Load the trained model
with zipfile.ZipFile('trained_gb_model.zip', 'r') as file:
    file.extract('trained_gb_model.joblib', path ='.')

model = joblib.load('trained_gb_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    movement_reactions = int(request.form['movement_reactions'])
    mentality_composure = int(request.form['mentality_composure'])
    passing = int(request.form['passing'])
    potential = int(request.form['potential'])
    release_clause_eur = int(request.form['release_clause_eur'])
    dribbling = int(request.form['dribbling'])
    wage_eur = int(request.form['wage_eur'])
    power_shot_power = int(request.form['power_shot_power'])
    value_eur = int(request.form['value_eur'])
    mentality_vision = int(request.form['mentality_vision'])
    attacking_short_passing = int(request.form['attacking_short_passing'])
    physic = int(request.form['physic'])
    skill_long_passing = int(request.form['skill_long_passing'])
    age = int(request.form['age'])
    shooting = int(request.form['shooting'])
    skill_ball_control = int(request.form['skill_ball_control'])
    real_face = int(request.form['real_face'])
    international_reputation = int(request.form['international_reputation'])
    skill_curve = int(request.form['skill_curve'])
    attacking_crossing = int(request.form['attacking_crossing'])
    power_long_shots = int(request.form['power_long_shots'])
    mentality_aggression = int(request.form['mentality_aggression'])
    

    # Create an input feature array from the user inputs
    input_features = np.array([
        movement_reactions, mentality_composure, passing, potential, release_clause_eur, dribbling, wage_eur,
        power_shot_power, value_eur, mentality_vision, attacking_short_passing, physic, skill_long_passing, age,
        shooting, skill_ball_control, real_face, international_reputation, skill_curve, attacking_crossing, power_long_shots,
        mentality_aggression
        # Add more features in the same order as used during training...
    ]).reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(input_features)

    # Calculate the confidence level (you can use any appropriate metric)
    confidence_level = calculate_confidence(input_features)  # Replace with your confidence calculation

    # Display the prediction and confidence level
    result = f'Predicted Rating: {prediction[0]:.2f}, Confidence Level: {confidence_level:.2f}'
    print(result)  # Print the result to the server console
    
    # Return the result as the response
    return result

def calculate_confidence(input_features):
    # Implement your confidence level calculation logic here
    # This can be based on prediction variance or any other relevant metric
    # For example, you can use the standard deviation of predictions
    
    predictions = model.predict(input_features)  # Make multiple predictions
    
    # Calculate the standard deviation of predictions
    prediction_std = np.std(predictions)
    
    # You can customize the confidence calculation logic based on your requirements
    # This is just a simple example
    
    confidence_level = 1.0 / (1.0 + prediction_std)  # Higher standard deviation results in lower confidence
    
    return confidence_level

@app.route('/')
def index():
    return render_template('prediction_form.html')

if __name__ == "__main__":
    app.run(debug=True)
    



