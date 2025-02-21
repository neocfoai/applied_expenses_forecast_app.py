from flask import Flask, jsonify, send_file
from flask_cors import cross_origin
import requests
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI

import base64
from io import BytesIO
from PIL import Image

import json
import datetime

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = JSONEncoder
company_id = 13288
class Config:
    ROOTIFY_API = "19a42cfe-170d-416a-8dab-cc8f227ae0b0"
    COMPANY_ID = "13288"
    OPENAI_API_KEY = "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"
    URL_BILL_PAYMENTS = "https://api.rootfi.dev/v3/accounting/bill_payments"
    STATIC_FOLDER = os.path.join(os.getcwd(), 'static')
    os.makedirs(STATIC_FOLDER, exist_ok=True)

class DataProcessor:
    @staticmethod
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
    
    

    @staticmethod
    def convert_to_ymd(date_column):
        return pd.to_datetime(date_column, errors='coerce').dt.date

    @staticmethod
    def process_sales_data():
        df = pd.json_normalize(DataProcessor.fetch_all_bill_payments(
            Config.ROOTIFY_API, Config.COMPANY_ID, Config.URL_BILL_PAYMENTS))
        
        columns_to_drop = [
            'rootfi_id', 'rootfi_created_at', 'rootfi_updated_at', 'rootfi_deleted_at',
            'rootfi_company_id', 'platform_id', 'platform_unique_id', 'bill_id',
            'credit_note_id', 'updated_at', 'document_number', 'currency_rate',
            'custom_fields', 'currency_id', 'payment_mode'
        ]
        
        df = df.drop(columns=columns_to_drop)
        df = df.rename(columns={
            'payment_id': 'Bill_Payment_ID',
            'account_id': 'ID',
            'contact_id': 'Vendor_Contact_ID',
            'amount': 'Payment_Amount',
            'memo': 'Payment_Memo',
            'payment_date': 'Bill_Payment_Date'
        })
        
        df['Bill_Payment_Date'] = DataProcessor.convert_to_ymd(df['Bill_Payment_Date'])
        return df
    
    @staticmethod
    def prepare_data_for_prophet(df, date_column, target_column):
        df = df.rename(columns={
            date_column: 'ds',
            target_column: 'y'
        })
        df = df.dropna(subset=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        # Add grouping by date
        df = df.groupby('ds', as_index=False).sum()
        return df

class ForecastGenerator:
    @staticmethod
    def prepare_data_for_prophet(df, date_column, target_column):
        df = df.rename(columns={
            date_column: 'ds',
            target_column: 'y'
        })
        df = df.dropna(subset=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        return df
        
# colors = ['#77C043', '#144944', '#C8FF96', '#45A49B', '#809191', '#0E2120', '#708B70']
    @staticmethod
    def fit_forecast_and_custom_plot(df, periods, freq='D', colors=None, save_path=None, name=None):
        if colors is None:
            colors = {'forecast': '#77C043', 'actual':'#144944', 'uncertainty': '#C8FF96'}

        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(df['ds'], df['y'], 'o', color=colors['actual'], alpha=0.6, label='Actual')
        ax.plot(forecast['ds'], forecast['yhat'], ls='-', color=colors['forecast'], label='Forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                       color=colors['uncertainty'], alpha=0.3, label='Uncertainty Interval')

        plt.title('Sales Analysis Forecast', size=15, pad=15)
        plt.xlabel('Date', size=12)
        plt.ylabel('Amount', size=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        if save_path:
            if name is None:
                name = "sales_analysis"
            
            forecast_path = os.path.join(save_path, f"{name}_forecast.png")
            plt.savefig(forecast_path, bbox_inches='tight', dpi=300)

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

@app.route('/<int:company_id>/expenses', methods=['GET'])
@cross_origin() 
def analyze_sales(company_id):
    try:
        sales_data = DataProcessor.process_sales_data()
        prophet_data = ForecastGenerator.prepare_data_for_prophet(
            sales_data, 'Bill_Payment_Date', 'Payment_Amount')
        
        # Change the name to match your URL
        forecast = ForecastGenerator.fit_forecast_and_custom_plot(
            prophet_data,
            periods=12,
            freq='M',
            save_path=Config.STATIC_FOLDER,
            name="expenses"  # Changed from "sales_analysis" to "expenses"
        )

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

        with open(os.path.join(Config.STATIC_FOLDER, "expenses_forecast.png"), "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this sales forecast image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }

        insights = vision_llm.invoke([system_message, user_message])

         # Prepare DataFrames for response
        forecast_df = pd.DataFrame({
            'date': forecast['ds'],
            'actual': prophet_data['y'].reindex(forecast.index),
            'predicted': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper']
        })
        
        # Components DataFrames
        trend_df = pd.DataFrame({
            'date': forecast['ds'],
            'trend': forecast['trend']
        })
        
        seasonal_components = {}
        for component in ['yearly', 'weekly', 'monthly']:
            if component in forecast.columns:
                seasonal_components[component] = pd.DataFrame({
                    'date': forecast['ds'],
                    component: forecast[component]
                })
        
        # Calculate some additional metrics
        total_actual = prophet_data['y'].sum()
        avg_actual = prophet_data['y'].mean()
        last_actual = prophet_data['y'].iloc[-1]
        forecast_total = forecast['yhat'][-12:].sum()  # Sum of next 12 months forecast
        
        # Prepare data for JSON response
        def prepare_df_for_json(df):
            return {
                'columns': df.columns.tolist(),
                'data': df.where(pd.notnull(df), None).values.tolist(),
                'index': df.index.tolist()
            }
        
        # Build response
        response_data = {
            'status': 'success',
            'insights': insights.content,
            'data': {
                'forecast': prepare_df_for_json(forecast_df),
                'components': {
                    'trend': prepare_df_for_json(trend_df)
                },
                'metrics': {
                    'total_expenses': float(total_actual),
                    'average_monthly_expenses': float(avg_actual),
                    'last_month_expenses': float(last_actual),
                    'forecasted_next_12_months': float(forecast_total),
                    'percent_change_forecast': float(((forecast_total - total_actual) / total_actual) * 100)
                }
            },
            'metadata': {
                'total_periods': len(forecast_df),
                'forecast_periods': 12,
                'frequency': 'Monthly',
                'start_date': forecast_df['date'].min().isoformat(),
                'end_date': forecast_df['date'].max().isoformat()
            }
        }
        
        # Add seasonal components if they exist
        for component, df in seasonal_components.items():
            response_data['data']['components'][component] = prepare_df_for_json(df)
        
        return jsonify(response_data)
        
        # return jsonify({
        #     'status': 'success',
        #     'insights': insights.content
        # })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/<int:company_id>/plots/<plot_name>', methods=['GET'])
@cross_origin() 
def get_plot(plot_name, company_id):
    try:
        # Add logging to debug file paths
        plot_path = os.path.join(Config.STATIC_FOLDER, plot_name)
        print(f"Attempting to access file at: {plot_path}")
        
        if not os.path.exists(plot_path):
            print(f"File not found at: {plot_path}")
            return jsonify({
                'status': 'error',
                'message': f'File not found: {plot_name}'
            }), 404
            
        return send_file(
            plot_path,
            mimetype='image/png'
        )
    except Exception as e:
        print(f"Error serving plot: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5003))
    app.run(host='0.0.0.0', port=port)
