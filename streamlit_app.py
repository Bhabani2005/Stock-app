import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
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

    df.describe()
    columns_to_drop = ['Adj Close']
    df.drop(columns=columns_to_drop, inplace = True)

    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')
    df.head(2)

    df.set_index('Date', inplace=True)

    df['Open'] = df['Open'].fillna(df['Open'].mean())  # Replace NaN with the column mean
    df['High'] = df['High'].fillna(df['High'].mean())
    df['Low'] = df['Low'].fillna(df['Low'].mean())
    df['Volume'] = df['Volume'].fillna(df['Volume'].mean())
    df['Close'] = df['Close'].fillna(df['Close'].mean())
   
    # Plot the 'Open' prices with the Date as the x-axis
    st.write("### Open Prices Over Time")

    # Plot the 'Open' prices over time using Plotly
    fig = px.line(df, y='Open')

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    # Feature selection
    st.write("### Feature Selection")
    feature_columns = st.multiselect("Select feature columns", df.columns.tolist(), default=df.columns[:4].tolist()) 
    target_column = st.selectbox("Select target column", df.columns.tolist(), index=len(df.columns)-1)

    # Data Preprocessing
    X = df[['Open','High','Low','Volume']]
    y = df['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    train_size = int(0.8 * len(df))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model Training
    svr_model = SVR(kernel='linear', C=50, epsilon=0.2)
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

    dframeS = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    # Display the first 5 rows in a more structured table
    st.markdown("### ITC Stock: Actual vs Predicted Values")
    st.write("Here's a comparison of the first 5 rows of the actual and predicted stock prices:")

    # Use st.dataframe for an interactive table
    st.dataframe(dframeS.head(50), width=700, height=200)

    # Optionally, use st.table for a static, cleaner look
    st.write("Static View of the Table:")
    #st.markdown("### Static View of the Table")
    st.table(dframeS.head(5))

   # Visualization
    graphS = dframeS.head(15)

    # Generate x-axis indices for the bars
    x_indices = np.arange(len(graphS))

    # Create a bar plot with offsets for Actual and Predicted
    fig, ax = plt.subplots(figsize=(12, 6))  # Create a Matplotlib figure

    # Bar width
    bar_width = 0.4

    # Plot Actual and Predicted with offsets
    actual_bars = ax.bar(x_indices - bar_width / 2, graphS['Actual'], label='Actual', color='blue', alpha=0.7, width=bar_width)
    predicted_bars = ax.bar(x_indices + bar_width / 2, graphS['Predicted'], label='Predicted', color='orange', alpha=0.7, width=bar_width)

    # Add titles and labels
    ax.set_title("ITC STOCK: Actual Price vs Predicted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(graphS.index, rotation=45)  # Rotate x-axis labels for better readability

    # Add legend with matching colors
    ax.legend(fontsize=12, loc='upper left')

    # Display the plot in Streamlit
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
