from flask import Flask, jsonify, send_file
from flask_cors import cross_origin
import requests
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from langchain_openai import ChatOpenAI
from PIL import Image

app = Flask(__name__)
company_id = 13288

# Configuration
class Config:
    ROOTIFY_API = "19a42cfe-170d-416a-8dab-cc8f227ae0b0"
    COMPANY_ID = "13288"
    OPENAI_API_KEY = "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"
    URL_BILL_PAYMENTS = "https://api.rootfi.dev/v3/accounting/bill_payments"
    URL_INVOICE_PAYMENTS = "https://api.rootfi.dev/v3/accounting/Invoice_Payments"

# Helper Functions
def fetch_all_bill_payments(api_key, company_id, base_url, limit=1000):
    headers = {"api_key": api_key}
    all_bill_payments = []
    next_cursor = None

    while True:
        params = {
            "limit": limit,
            "rootfi_company_id[eq]": company_id
        }
        if next_cursor:
            params["next"] = next_cursor

        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        all_bill_payments.extend(data["data"])

        if not data.get("next"):
            break

        next_cursor = data["next"]

    return all_bill_payments

# Function to convert to 'YYYY-MM-DD' format
def convert_to_ymd(date_column):
    """
    Converts ISO 8601 dates to 'YYYY-MM-DD' format.

    Parameters:
        date_column (pd.Series): A pandas Series containing ISO 8601 date strings.

    Returns:
        pd.Series: A pandas Series with dates in 'YYYY-MM-DD' format.
    """
    return pd.to_datetime(date_column, errors='coerce').dt.date



def prepare_data_for_prophet(df, date_column, target_column):
    df = df.rename(columns={
        date_column: 'ds',
        target_column: 'y'
    })
    df = df.dropna(subset=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df

# colors = ['#77C043', '#144944', '#C8FF96', '#45A49B', '#809191', '#0E2120', '#708B70']
def fit_forecast_and_custom_plot(df, periods, freq='D', colors=None, save_path=None, name=None):
    if colors is None:
        colors = {
            'forecast': '#77C043',
            'actual': '#144944',
            'uncertainty': '#C8FF96'
        }

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Create forecast plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(df['ds'], df['y'], 'o', color=colors['actual'], alpha=0.6, label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], ls='-', color=colors['forecast'], label='Forecast')
    ax.fill_between(forecast['ds'],
                    forecast['yhat_lower'],
                    forecast['yhat_upper'],
                    color=colors['uncertainty'],
                    alpha=0.3,
                    label='Uncertainty Interval')

    plt.title('Time Series Forecast', size=15, pad=15)
    plt.xlabel('Date', size=12)
    plt.ylabel('Value', size=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save plots if save_path is provided
    if save_path:
        if name is None:
            name = "forecast"
        
        # Save forecast plot
        forecast_path = os.path.join(save_path, f"{name}_forecast.png")
        plt.savefig(forecast_path, bbox_inches='tight', dpi=300)

        # Generate and save components plot
        components_fig = model.plot_components(forecast)
        for ax in components_fig.axes:
            ax.set_facecolor('#F8F9FA')
            ax.grid(True, alpha=0.3)
            for line in ax.lines:
                line.set_color('#2C3E50')
            for collection in ax.collections:
                collection.set_alpha(0.3)
                collection.set_facecolor('#AED6F1')

        components_path = os.path.join(save_path, f"{name}_components.png")
        components_fig.savefig(components_path, bbox_inches='tight', dpi=300)

    return forecast

def process_payments_data():
    # Fetch data
    df_bill = pd.json_normalize(fetch_all_bill_payments(Config.ROOTIFY_API, Config.COMPANY_ID, Config.URL_BILL_PAYMENTS))
    df_invoice = pd.json_normalize(fetch_all_bill_payments(Config.ROOTIFY_API, Config.COMPANY_ID, Config.URL_INVOICE_PAYMENTS))

    # Process bill payments
    columns_to_drop_bill = [
        'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
        'rootfi_company_id', 'platform_id', 'platform_unique_id', 'bill_id',
        'credit_note_id', 'updated_at', 'document_number', 'currency_rate',
        'custom_fields', 'currency_id', 'payment_mode'
    ]
    df_bill = df_bill.drop(columns=columns_to_drop_bill)
    df_bill = df_bill.rename(columns={
        'payment_id': 'Bill_Payment_ID',
        'account_id': 'ID',
        'contact_id': 'Vendor_Contact_ID',
        'amount': 'Payment_Amount',
        'memo': 'Payment_Memo',
        'payment_date': 'Bill_Payment_Date'
    })

    # Process invoice payments
    columns_to_drop_invoice = [
        'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
        'rootfi_company_id', 'platform_id', 'platform_unique_id', 'invoice_id',
        'credit_note_id', 'updated_at', 'document_number', 'currency_rate',
        'custom_fields', 'currency_id', 'payment_mode'
    ]
    df_invoice = df_invoice.drop(columns=columns_to_drop_invoice)
    df_invoice = df_invoice.rename(columns={
        'payment_id': 'Invoice_Payment_ID',
        'account_id': 'ID',
        'contact_id': 'Vendor_Contact_ID',
        'amount': 'Payment_Amount',
        'memo': 'Payment_Memo',
        'payment_date': 'Invoice_Payment_Date'
    })

    # Convert dates
    df_bill['Bill_Payment_Date'] = convert_to_ymd(df_bill['Bill_Payment_Date'])
    df_invoice['Invoice_Payment_Date'] = convert_to_ymd(df_invoice['Invoice_Payment_Date'])

    # Prepare data for analysis
    df_bill_grouped = df_bill.groupby('Bill_Payment_Date')['Payment_Amount'].sum().reset_index()
    df_invoice_grouped = df_invoice.groupby('Invoice_Payment_Date')['Payment_Amount'].sum().reset_index()

    # Merge dataframes
    merged_df = pd.merge(
        df_invoice_grouped, 
        df_bill_grouped, 
        left_on='Invoice_Payment_Date', 
        right_on='Bill_Payment_Date', 
        how='outer',
        suffixes=('_invoice', '_bill')
    )
    
    # Fill NaN in payment amounts with 0 (numeric columns only)
    merged_df['Payment_Amount_invoice'] = merged_df['Payment_Amount_invoice'].fillna(0)
    merged_df['Payment_Amount_bill'] = merged_df['Payment_Amount_bill'].fillna(0)

    # Create date column by combining both date sources
    merged_df['date'] = merged_df['Invoice_Payment_Date'].combine_first(merged_df['Bill_Payment_Date'])

    # Calculate net amount
    merged_df['net_amount'] = merged_df['Payment_Amount_invoice'] - merged_df['Payment_Amount_bill']

    # Prepare final dataframe
    final_df = merged_df[['date', 'Payment_Amount_invoice', 'Payment_Amount_bill', 'net_amount']]
    final_df = final_df.rename(columns={
        'Payment_Amount_invoice': 'invoice_payments',
        'Payment_Amount_bill': 'bill_payments'
    }).sort_values('date')

    return final_df

# Flask Routes
@app.route('/<int:company_id>/generate_forecast', methods=['GET'])
@cross_origin() 
def generate_forecast(company_id):
    try:
        # Process data
        cashflow_data = process_payments_data()
        cashflow_data = prepare_data_for_prophet(cashflow_data, 'ds', 'net_amount')

        # Generate forecast
        save_path = os.path.join(os.getcwd(), 'static')
        os.makedirs(save_path, exist_ok=True)
        
        forecast = fit_forecast_and_custom_plot(
            cashflow_data,
            periods=12,
            freq='M',
            save_path=save_path,
            name="Cash_Flow"
        )

        # Generate insights using OpenAI
        vision_llm = ChatOpenAI(
            model='gpt-4o',
            temperature=0.2,
            api_key=Config.OPENAI_API_KEY
        )

        SYSTEM_PROMPT = f"""
    You are provided with a graph generated using the Facebook Prophet model for time series forecasting. Please carefully analyze the graph and extract the following information:

    - Graph Title:
        - If the graph includes a title, please state it exactly as shown.
        - If the graph does not have a title, create a descriptive and relevant title based on the graph's content, time frame, and key trends.

    - Time Period of the Data:
        - Identify the exact time range covered by the data. This could include specific start and end dates, months, quarters, or years. If the time period is unclear, provide the most reasonable estimate based on visible markers on the graph.

    - Key Features or Components:
        - Break down the primary components visible in the graph, such as:
            - **Trend**: The long-term movement in the data (e.g., increasing or decreasing trend).
            - **Seasonality**: Any recurring patterns or cycles in the data (e.g., yearly, weekly, daily cycles).
            - **Forecast**: Note the forecasted data points and any confidence intervals shown.
        - Explain any correlations or relationships that can be inferred from the graph’s features.

    - Anomalies:
        - Identify and describe any unusual deviations or outliers in the graph that do not align with the expected trends or patterns. These may include:
            - Sudden spikes or drops in the data.
            - Patterns that deviate from seasonal expectations.
            - Irregularities that stand out as potential errors or unusual events.

    - Detailed Description of the Graph:
        - Provide a detailed and thorough interpretation of the graph, including:
            - A description of the data being presented (e.g., sales, stock prices, temperature).
            - The overall trend and any significant fluctuations observed.
            - A breakdown of any seasonal or cyclical patterns present.
            - How the forecast is depicted in the graph (e.g., shaded areas for confidence intervals).
            - The reliability and potential accuracy of the model's predictions, based on the visual representation of the graph.
            - Find numerical insights as well such as percent change, growth rate etc
        - Discuss what insights can be drawn from the graph about the future behavior of the time series, based on trends, seasonality, and anomalies observed.



    - Recommendations:
        - Based on the analysis, provide any recommendations or next steps. These could include:
            - Adjustments or improvements to the model (if applicable).
            - Potential actions based on the forecast (e.g., planning for expected changes, addressing anomalies).
            - Suggestions for future data collection or forecasting.
"""

        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        }

        image_path = os.path.join(save_path, "Cash_Flow_components")
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me data insights from this image "},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }

        response = vision_llm.invoke([system_message, user_message])

        return jsonify({
            'status': 'success',
            'insights': response.content  # Only return the insights
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def generate_forecast_data(data_df, periods=12, freq='M'):
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(data_df['date']),
        'y': data_df['net_amount']
    })
    
    # Fit Prophet model
    model = Prophet()
    model.fit(prophet_df)
    
    # Generate future dates
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Make forecast
    forecast = model.predict(future)
    
    # Prepare forecast dataframe
    forecast_df = pd.DataFrame({
        'date': forecast['ds'],
        'predicted_value': forecast['yhat'],
        'lower_bound': forecast['yhat_lower'],
        'upper_bound': forecast['yhat_upper']
    })
    
    return forecast_df

@app.route('/<int:company_id>/get_data', methods=['GET'])
@cross_origin()
def get_data(company_id):
    try:
        # Get historical data
        historical_data = process_payments_data()
        
        # Generate forecast
        forecast_data = generate_forecast_data(historical_data)
        
        # Convert to dictionary format for JSON response
        historical_dict = historical_data.to_dict(orient='records')
        forecast_dict = forecast_data.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'data': {
                'historical_data': historical_dict,
                'forecast_data': forecast_dict
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/<int:company_id>/plots/<plot_name>', methods=['GET'])
@cross_origin() 
def get_plot(plot_name, company_id):
    try:
        return send_file(
            os.path.join('static', plot_name),
            mimetype='image/png'
        )
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404

if __name__ == '__main__':
    # Render sets the PORT environment variable automatically
    port = int(os.getenv('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)