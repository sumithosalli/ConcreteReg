import os
import zipfile
import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error
import lightgbm 
from lightgbm import LGBMRegressor,early_stopping


def Load_dataset(path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve zip path relative to this script when a relative path is provided
    zip_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
    # If resolution didn't find the file, also try the basename in the script dir
    if not os.path.exists(zip_path):
        alt = os.path.join(base_dir, os.path.basename(path))
        if os.path.exists(alt):
            zip_path = alt
        else:
            raise FileNotFoundError(f"Zip file not found at {path!r} (tried {zip_path!r} and {alt!r})")
    extract_dir = os.path.join(base_dir, "data")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    # Expect Concrete_Data.csv in the extracted files; try to locate any CSV if name differs
    csv_path = os.path.join(extract_dir, "Concrete_Data.csv")
    if not os.path.exists(csv_path):
        # search for first csv inside extract_dir
        found = None
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith('.csv'):
                    found = os.path.join(root, f)
                    break
            if found:
                break
        if found:
            csv_path = found
        else:
            raise FileNotFoundError(f"Concrete_Data.csv not found after extracting {zip_path!r} into {extract_dir!r}")
    return pd.read_csv(csv_path)

#adding new features
def add_features(X):
    X2 = X.copy()
    # replace these with features that make sense for your dataset
    if {'cement','water'}.issubset(X2.columns):
        X2['cement_to_water'] = X2['cement'] / (X2['water'] + 1e-9)
    if {'coarse_aggregate','fine_aggregate'}.issubset(X2.columns):
        X2['agg_ratio'] = X2['coarse_aggregate'] / (X2['fine_aggregate'] + 1e-9)
    if 'age' in X2.columns and 'cement' in X2.columns:
        X2['cement_times_age'] = X2['cement'] * X2['age']
    return X2  

#define loss function
def compute_loss(test_labels,preds):
    mae=mean_absolute_error(test_labels,preds)
    mse=mean_squared_error(test_labels,preds)
    rmse=root_mean_squared_error(test_labels,preds)
    r2=r2_score(test_labels,preds)
    return{
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
    
#split data into train and test data    
def split_data(dataset,input_cols,output_cols):
    train_data,test_data=train_test_split(dataset,test_size=0.05,random_state=42)
    train_inputs=train_data[input_cols].copy()
    train_target=train_data[output_cols].copy()
    test_inputs=test_data[input_cols].copy()
    test_target=test_data[output_cols].copy()
    return train_inputs,train_target,test_inputs,test_target

#adding min max scaler
def min_max_scaling(train_inputs,test_inputs):
    scaler=MinMaxScaler()
    scaler.fit(train_inputs)
    train_inputs_scaled=scaler.transform(train_inputs)
    test_inputs_scaled=scaler.transform(test_inputs)
    return train_inputs_scaled,test_inputs_scaled

#printing the loss metrics
def print_metrics(test_target,predictions):
    metrics=compute_loss(test_target,predictions)
    for name,value in metrics.items():
        print(f"{name}: {value:.4f}\n")
        
#function to predict single instance 
def predict_single(cement, blast_furnace_slag, fly_ash, water,
                   superplasticizer, coarse_aggregate,
                   age,cement_to_water,cement_times_age,scaler_obj, model_obj):
        cols = ["cement", "blast_furnace_slag", "fly_ash", "water",
                "superplasticizer", "coarse_aggregate", "age","cement_to_water"
                ,"cement_times_age"]
        df = pd.DataFrame([[cement, blast_furnace_slag, fly_ash, water,
                            superplasticizer, coarse_aggregate,
                            age,cement_to_water,cement_times_age]],columns=cols)
        scaled = scaler_obj.transform(df)
        pred = model_obj.predict(scaled)
        return float(pred[0])
    
#making a new pewdiction by taking user input
def get_user_input_and_predict(scaler_obj, model_obj):
    try:
        print("Enter values for prediction:")
        cement = float(input("Cement: "))
        blast_furnace_slag = float(input("Blast Furnace Slag: "))
        fly_ash = float(input("Fly Ash: "))
        water = float(input("Water: "))
        superplasticizer = float(input("Superplasticizer: "))
        coarse_aggregate = float(input("Coarse Aggregate: "))
        age = float(input("Age: "))
        # Derived features
        cement_to_water = cement / (water + 1e-9)
        cement_times_age = cement * age
        # ✅ FIXED: Removed fine_aggregate argument
        result = predict_single(
            cement, blast_furnace_slag, fly_ash, water,
            superplasticizer, coarse_aggregate, age,
            cement_to_water, cement_times_age,
            scaler_obj, model_obj
        )
        print(f"Predicted concrete strength: {result:.4f}")
    except Exception as e:
        print("Error in input or prediction:", e)

        
if __name__ == "__main__":
    df = Load_dataset("archive.zip")
    print("Loaded dataset with shape:", df.shape)
    # Clean up column names first (remove spaces)
    df.columns = df.columns.str.strip()
    # Now safely drop fine_aggregate if it exists
    if "fine_aggregate" in df.columns:
        df = df.drop("fine_aggregate", axis=1)
    else:
        print("⚠️ 'fine_aggregate' column not found — skipping drop.")
    #dropping fine_aggregate column
    dataset = add_features(df)
    print("\n",dataset.columns)
    print("\n Dataset after feature engineering: ", dataset.shape)
    target_col = 'concrete_compressive_strength'
    input_cols = [c for c in dataset.columns if c != target_col]
    X_train, y_train, X_test, y_test = split_data(dataset, input_cols, target_col)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = lightgbm.LGBMRegressor(
        objective='regression',
        learning_rate=0.01,
        n_estimators=5000,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=True)]
    )
    preds = model.predict(X_test_scaled)
    metrics = compute_loss(y_test, preds)
    print("Model metrics:", metrics)
    joblib.dump(model, 'concrete_strength_model.pkl')
    print("\n✅ Model trained and saved successfully!")  
    get_user_input_and_predict(scaler, model)