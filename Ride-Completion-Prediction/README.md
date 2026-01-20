# 🚕 Predicting Ride Completion in NCR Bookings

### 🎯 Objective
Predict the probability of ride completion to reduce cancellations, improve dispatch efficiency, and minimize operational risk.

### 🛠 Tools
Python, Pandas, NumPy, Scikit-learn, LightGBM, Feature Engineering

### 🔍 Approach
Engineered temporal, geographic, and historical behavioral features while preventing data leakage. Applied time-based splitting and handled class imbalance for robust modeling.

### ⭐ Key Insights
- Historical customer cancellation behavior strongly impacts completion probability
- Peak-hour and low-supply zones show higher non-completion risk
- LightGBM delivered best performance across ROC-AUC and F1-score
