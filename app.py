
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("decision_tree_model.pkl")

st.title("☕ Starbucks Coffee Calories Predictor")
st.write("Select a coffee to see the model's calorie prediction.")

coffee_options = [
    "Caffe Americano",
    "Cappuccino",
    "Caffe Latte",
    "Caramel Macchiato",
    "Mocha"
]

coffee = st.selectbox("Choose your coffee", coffee_options)

if st.button("Predict Calories"):
    # Prepare input using the same features as training
    input_df = pd.DataFrame({"Coffee": [coffee]})
    input_encoded = pd.get_dummies(input_df)

    # Match the model's original training columns
    if hasattr(model, "feature_names_in_"):
        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    prediction = model.predict(input_encoded)[0]

    st.success(f"Predicted calories: {prediction:.1f}")
    st.caption(
        "This is a learning demo using a small sample dataset, "
        "not official Starbucks nutrition information."
    )
