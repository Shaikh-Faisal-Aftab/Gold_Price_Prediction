import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Streamlit app
def main():
    st.title("Gold Price Prediction")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Preprocess the data
        data.date = pd.to_datetime(data.date).reset_index(drop=True)
        daily_data = data.copy()
        daily_data['e1'] = daily_data['price'].shift(-1)
        daily_data['e2'] = daily_data['price'].shift(-2)
        daily_data['e3'] = daily_data['price'].shift(-3)
        daily_data['e4'] = daily_data['price'].shift(-4)
        daily_data['e5'] = daily_data['price'].shift(-5)
        daily_data['e6'] = daily_data['price'].shift(-6)
        daily_data['e7'] = daily_data['price'].shift(-7)
        daily_data = daily_data.dropna()

        x_features = daily_data[['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']].values
        y_target = daily_data['price'].values

        X_train, X_test = x_features[:2000], x_features[2000:]
        y_train, y_test = y_target[:2000], y_target[2000:]

        # Train models
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)

        ran_model = RandomForestRegressor(n_estimators=142)
        ran_model.fit(X_train, y_train)

        xg_model = XGBRegressor()
        xg_model.fit(X_train, y_train)

        # Get user input
        prediction_days = st.slider("Select number of days for prediction:", min_value=1, max_value=45, value=7)

        # Make predictions
        lin_predictions = lin_model.predict(X_test)
        ran_predictions = ran_model.predict(X_test)
        xg_predictions = xg_model.predict(X_test)

        # Calculate RMSE
        lin_rmse = sqrt(mean_squared_error(y_test, lin_predictions))
        ran_rmse = sqrt(mean_squared_error(y_test, ran_predictions))
        xg_rmse = sqrt(mean_squared_error(y_test, xg_predictions))

        # Display RMSE values
        st.subheader("RMSE Values:")
        st.write("Linear Regression RMSE:", lin_rmse)
        st.write("Random Forest RMSE:", ran_rmse)
        st.write("XGBoost RMSE:", xg_rmse)

        # Determine the best model based on RMSE
        best_model = min(lin_rmse, ran_rmse, xg_rmse)
        if best_model == lin_rmse:
            st.write("Best Model: Linear Regression")
        elif best_model == ran_rmse:
            st.write("Best Model: Random Forest")
        else:
            st.write("Best Model: XGBoost")

        # Plot last 100 days of data and future predictions
        fig = go.Figure()
        last_30_days = data[-60:]
        fig.add_trace(go.Scatter(x=last_30_days.date, y=last_30_days.price, mode='lines', name='Last 60 Days'))
        future_dates = pd.date_range(start=last_30_days.date.iloc[-1], periods=prediction_days, freq='D')
        future_pred = []
        z = X_test[-1].tolist()
        for i in range(prediction_days):
            r = np.array(z[-7:]).reshape(1, -1)
            ranf_f = lin_model.predict(r)
            z.extend(ranf_f.tolist())
            future_pred.append(ranf_f[0])
        fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode='lines', name='Future Predictions'))
        fig.update_layout(title="Gold Price Prediction",
                          xaxis_title="Date",
                          yaxis_title="Price",
                          width=900,
                          height=600)
        st.plotly_chart(fig)

# Run the Streamlit app
if __name__ == "__main__":
    main()
