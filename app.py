import streamlit as st
import joblib  # For loading your trained model
import numpy as np  # For data manipulation

# Load the trained model
model = joblib.load('trained_model.joblib')
# Streamlit app
st.title('Player Rating Predictor')

# Input widgets for the features used during training
movement_reactions = st.slider('Movement Reactions', min_value=0, max_value=100, value=50)
mentality_composure = st.slider('Mentality Composure', min_value=0, max_value=100, value=50)
passing = st.slider('Passing', min_value=0, max_value=100, value=50)
potential = st.slider('Potential', min_value=0, max_value=100, value=50)
release_clause_eur = st.slider('Release Clause EUR', min_value=0, max_value=100000000, value=50000000)
dribbling = st.slider('Dribbling', min_value=0, max_value=100, value=50)
wage_eur = st.slider('Wage EUR', min_value=0, max_value=1000000, value=500000)
power_shot_power = st.slider('Power Shot Power', min_value=0, max_value=100, value=50)
value_eur = st.slider('Value EUR', min_value=0, max_value=100000000, value=50000000)
mentality_vision = st.slider('Mentality Vision', min_value=0, max_value=100, value=50)
attacking_short_passing = st.slider('Attacking Short Passing', min_value=0, max_value=100, value=50)
physic = st.slider('Physic', min_value=0, max_value=100, value=50)
skill_long_passing = st.slider('Skill Long Passing', min_value=0, max_value=100, value=50)
age = st.slider('Age', min_value=0, max_value=50, value=25)
shooting = st.slider('Shooting', min_value=0, max_value=100, value=50)
skill_ball_control = st.slider('Skill Ball Control', min_value=0, max_value=100, value=50)
real_face = st.checkbox('Real Face')
international_reputation = st.slider('International Reputation', min_value=0, max_value=5, value=2)
skill_curve = st.slider('Skill Curve', min_value=0, max_value=100, value=50)
attacking_crossing = st.slider('Attacking Crossing', min_value=0, max_value=100, value=50)
power_long_shots = st.slider('Power Long Shots', min_value=0, max_value=100, value=50)
mentality_aggression = st.slider('Mentality Aggression', min_value=0, max_value=100, value=50)

# Create an input feature array from the user inputs
input_features = np.array([
    movement_reactions, mentality_composure, passing, potential, release_clause_eur, dribbling, wage_eur,
    power_shot_power, value_eur, mentality_vision, attacking_short_passing, physic, skill_long_passing, age,
    shooting, skill_ball_control, int(real_face), international_reputation, skill_curve, attacking_crossing, power_long_shots,
    mentality_aggression
]).reshape(1, -1)

# Make predictions using the model
prediction = model.predict(input_features)

# Display the prediction
st.write(f'Predicted Rating: {prediction[0]:.2f}')

if __name__ == "__main__":
    st.run()

