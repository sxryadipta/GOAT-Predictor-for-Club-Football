import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


data = pd.read_csv("data/players.csv")


features = [
    "club_goals",
    "club_assists",
    "ucl_goals",
    "ucl_assists",
    "league_titles",
    "ucl_titles",
    "domestic_cups",
    "ballon_dors",
    "total_club_trophies",
    "positional_fluidity",
    "team_versatility"
]


train_data = data.dropna(subset=["goat_score"])
X_train = train_data[features]
y_train = train_data["goat_score"]


predict_data = data[data["goat_score"].isna()]
X_predict = predict_data[features]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_predict_scaled = scaler.transform(X_predict)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


predictions = model.predict(X_predict_scaled)


data.loc[data["goat_score"].isna(), "goat_score"] = np.round(predictions, 2)


print("\n=== GOAT Predictions ===")
for _, row in data.iterrows():
    print(f"{row['name']}: {row['goat_score']}")


data.to_csv("data/players_with_predictions.csv", index=False)
print("\nPredictions saved to 'data/players_with_predictions.csv'")
