import mlflow

mlflow.set_experiment("Rossmann_Sales_Forecasting")
with mlflow.start_run(run_name="RandomForest_Validation"):
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 60)
    mlflow.log_param("max_depth", 18)
    mlflow.log_param("validation", "last 6 weeks")
    mlflow.log_metric("RMSPE", 0.2425)
    mlflow.log_metric("MAE", 1057.84)
    mlflow.log_metric("RMSE", 1529.85)
    print("MLflow run logged.")
