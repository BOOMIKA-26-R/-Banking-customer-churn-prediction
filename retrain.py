import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def retrain_model(data_path):
    df = pd.read_csv(data_path)
    
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Successfully retrained and updated model and scaler!")

if __name__ == "__main__":
    retrain_model('new_bank_data.csv')
