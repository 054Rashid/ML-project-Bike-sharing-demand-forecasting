ML Project: Bike Sharing Demand Forecasting

This project tackles a time-series regression problem from the Kaggle Bike Sharing Demand Competition : https://www.kaggle.com/c/bike-sharing-demand
The goal is to forecast bike rental counts based on environmental and temporal features.
We use feature engineering, EDA, and a Gradient Boosting Regressor to build a powerful model.


📑 Project Overview
Objective: Predict bike rental counts (continuous values) given historical data.
Core Challenge: Extracting and engineering features from the datetime column to capture temporal patterns.
Metric: RMSLE (Root Mean Squared Logarithmic Error).
Model Used: GradientBoostingRegressor (tree-based ensemble model).

🗂️ Workflow

Load & Parse Data
Feature Engineering from datetime (year, month, hour, weekday, etc.)
EDA: Visualize demand patterns over time, weather, and weekdays.
Preprocess Data: Log-transform skewed target variable count.
Train Model: Gradient Boosting Regressor with tuned parameters.
Evaluate Model: Compute RMSLE on a validation set.
Feature Importance: Identify key drivers of demand.
Make Predictions: Generate Kaggle submission file.

📁 Project Structurebike-sharing-demand/
│
├── train.csv                # Training data (from Kaggle)
├── test.csv                 # Test data (from Kaggle)
├── bike_sharing_demand.py   # Main Python script
└── README.md                # Project documentation

⚙️ Requirements
Install dependencies via pip: pip install pandas numpy scikit-learn matplotlib seaborn

📊 Key Features Engineered

Year
Month
Day
Hour
Day of Week
Month Name

📈 Exploratory Data Analysis

The script visualizes:
Demand by Hour of the Day
Demand by Day of the Week
Demand by Weather Condition
These plots reveal commuting peaks, weekday vs. weekend trends, and weather impact.

🔧 Model & Evaluation
Model: Gradient Boosting Regressor
Hyperparameters: n_estimators=500, learning_rate=0.05, max_depth=4
Target Transformation: Log-transform of count to reduce skewness.
Metric: RMSLE on validation set.

🧠 Feature Importance

The model outputs the top 10 most important features driving bike demand.
This helps understand which temporal or weather factors matter most.

📝 Output
submission.csv
Contains datetime and predicted bike counts.
Submit directly to Kaggle to get your public leaderboard score.

🔮 Future Improvements
Use time-based cross-validation instead of random splits.
Try XGBoost or LightGBM for faster training and potentially higher accuracy.
Add holiday/event features to capture spikes or dips.
