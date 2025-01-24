def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Encode categorical variables if any
    df = pd.get_dummies(df, drop_first=True)
    
    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def load_and_preprocess_data(file_path):
    df = load_data(file_path)
    return preprocess_data(df)