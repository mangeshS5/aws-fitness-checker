# fitness_app.py
import streamlit as st
import pickle
import pandas as pd

# Load the trained model and label encoder
with open("fitness_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# Streamlit app layout
def main():
    st.set_page_config(page_title="Fitness Checker", layout="centered")
    st.title("🏋️‍♂️ Fitness Level Prediction App")

   

    # User input form
    age = st.number_input("Enter your Age:", min_value=0, max_value=100, step=1)
    bmi = st.number_input("Enter your BMI:", min_value=10.0, max_value=50.0, step=0.1, format="%.1f")
    exercise = st.number_input("Exercise Frequency (days/week):", min_value=0, max_value=7, step=1)
    heart_rate = st.number_input("Resting Heart Rate:", min_value=40, max_value=200, step=1)

    # Predict button
    if st.button("💡 Predict Fitness Status"):
        try:
            # Prepare input data
            input_data = pd.DataFrame([[age, bmi, exercise, heart_rate]],
                                      columns=["Age", "BMI", "ExerciseFreq", "HeartRate"])
            
            # Predict using the model
            prediction = model.predict(input_data)[0]
            result = le.inverse_transform([prediction])[0]

            # Display result
            st.success(f"✅ You are **{result}**!")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

if __name__ == '__main__':
    main()
