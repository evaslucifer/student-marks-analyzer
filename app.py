import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page title
st.title("Student Marks Analyzer")

# Upload CSV
uploaded_file = st.file_uploader("Upload Student Dataset", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Show statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Quick metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Average Marks", round(df["Marks"].mean(), 2))
    col2.metric("Highest Marks", round(df["Marks"].max(), 2))
    col3.metric("Lowest Marks", round(df["Marks"].min(), 2))

    # Scatter plot
    st.subheader("Study Time vs Marks")

    fig, ax = plt.subplots()
    ax.scatter(df["time_study"], df["Marks"])
    ax.set_xlabel("Study Time")
    ax.set_ylabel("Marks")

    st.pyplot(fig)

    # Correlation
    st.subheader("Correlation Matrix")
    st.write(df.corr())

    # Train model
    X = df[["time_study"]]
    y = df["Marks"]

    model = LinearRegression()
    model.fit(X, y)

    # Prediction section
    st.subheader("Predict Marks")

    hours = st.slider(
        "Select Study Hours",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    prediction = model.predict(
        pd.DataFrame([[hours]], columns=["time_study"])
    )

    st.success(f"Predicted Marks: {prediction[0]:.2f}")

    # Regression line
    st.subheader("Regression Graph")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["time_study"], df["Marks"])
    ax2.plot(df["time_study"], model.predict(X))

    ax2.set_xlabel("Study Time")
    ax2.set_ylabel("Marks")

    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file.")