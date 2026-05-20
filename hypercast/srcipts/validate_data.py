# scripts/validate_data.py
def validate_ercot_data(df):
    """Basic validation checks"""
    checks = {
        'no_nulls_in_price': df['price'].isnull().sum() == 0,
        'no_negative_prices': (df['price'] >= -500).all(),  # Allow some negative
        'complete_time_series': len(df) > 0,
        'valid_date_range': df['timestamp'].min() < df['timestamp'].max()
    }
    
    for check, passed in checks.items():
        print(f"{check}: {'✓' if passed else '✗'}")
    
    return all(checks.values())