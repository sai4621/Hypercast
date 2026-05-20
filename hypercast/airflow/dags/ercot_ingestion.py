# airflow/dags/ercot_ingestion.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def fetch_daily_data(**context):
    # Import your fetch function
    from scripts.fetch_ercot_data import fetch_ercot_prices
    execution_date = context['execution_date']
    return fetch_ercot_prices(execution_date, execution_date)

def validate_data(**context):
    from scripts.validate_data import validate_ercot_data
    import pandas as pd
    
    # Get filename from previous task
    ti = context['ti']
    filename = ti.xcom_pull(task_ids='fetch_data')
    
    df = pd.read_csv(filename)
    is_valid = validate_ercot_data(df)
    
    if not is_valid:
        raise ValueError("Data validation failed!")

default_args = {
    'owner': 'hypercast',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ercot_daily_ingestion',
    default_args=default_args,
    description='Fetch daily ERCOT prices',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    fetch_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_daily_data
    )
    
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data
    )
    
    fetch_task >> validate_task