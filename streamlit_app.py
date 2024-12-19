import streamlit as st
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Title
st.title('Stock Price Prediction App')

# Sidebar for uploading data
st.sidebar.header('Upload Dataset')
uploaded_file = st.sidebar.file_uploader("ITC.NS.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Feature selection
    st.write("### Feature Selection")
    feature_columns = st.multiselect("Select feature columns", df.columns.tolist(), default=df.columns[:-1])
    target_column = st.selectbox("Select target column", df.columns.tolist(), index=len(df.columns)-1)

    # Data Preprocessing
    X = df['Open','High','Low','Volume']
    y = df['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    train_size = int(0.8 * len(df))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model Training
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = svr_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)

    st.write("### Model Evaluation")
    st.write(f"R-squared: {r2:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Root Mean Squared Error: {rmse:.4f}")

    # Visualization
    st.write("### Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.reset_index(drop=True), label='Actual', marker='o')
    ax.plot(y_pred, label='Predicted', marker='x')
    ax.legend()
    st.pyplot(fig)

    # User Input for Prediction
    st.sidebar.header('Predict Stock Price')
    input_data = []
    for col in feature_columns:
        value = st.sidebar.number_input(f"Input {col}", value=float(df[col].mean()))
        input_data.append(value)

    if st.sidebar.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = svr_model.predict(input_scaled)
        st.sidebar.write(f"Predicted Stock Price: {prediction[0]:.2f}")
