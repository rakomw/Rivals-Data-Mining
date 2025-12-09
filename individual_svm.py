import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from typing import List, Dict, Any
import re
import os
import json
import numpy as np
from typing import List, Dict, Tuple

def load_individual_data(data_path: str = "individual_player_stats.json") -> Dict:
    """Load JSON file contained individual player statistics per match."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run cleaning.py first.")
    print(f"[INFO] Loading cleaned matches from {data_path}")
    with open(data_path, "r") as f:
        return json.load(f)

def safe_float(x) -> float:
    """Safe conversion to float."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        # Remove common formatting like commas and percent signs
        s = str(x).replace(",", "").replace("%", "").strip()
        return float(s) if s else 0.0
    except:
        return 0.0

def parse_minutes(time_str: str) -> float:
    """Convert '8.5m' or '30s' to minutes."""
    if not isinstance(time_str, str):
        return 0.0
    t = time_str.lower().strip()

    if t.endswith("m"):
        try:
            return float(t[:-1])
        except:
            return 0.0

    if t.endswith("s"):
        try:
            sec = float(t[:-1])
            return round(sec / 60.0, 4)
        except:
            return 0.0

    try:
        return float(t)
    except:
        return 0.0

def average_stats_per_player(individual_data: Dict) -> pd.DataFrame:
    all_players_data = []
    for player_id in individual_data.keys():
        player_match_data = {}
        num_matches = 0
        for match in individual_data[player_id]:
            match_duration = 0.0
            for hero in match.get("heroes_played", []):
                match_duration += parse_minutes(hero[1])
            if match_duration == 0.0: continue
            if safe_float(match.get("rank", 0)) <= 0: continue
            num_matches += 1
            for key in match.keys():
                if key == "player_id" or key == "py/object" or key == "heroes_played": continue
                if key not in player_match_data:
                    player_match_data[key] = 0
                player_match_data[key] += safe_float(match[key])
        for key in player_match_data.keys():
            player_match_data[key] /= num_matches
            player_match_data[key] = round(player_match_data[key], 3)
        if len(player_match_data) > 0:
            all_players_data.append(player_match_data)
    return pd.DataFrame(all_players_data)

def build_dataframe(individual_data: Dict) -> pd.DataFrame:
    all_players_data = []
    for player_id in individual_data.keys():
        player_match_data = []
        for match in individual_data[player_id]:
            match_stats = {}
            match_duration = 0.0
            for hero in match.get("heroes_played", []):
                match_duration += parse_minutes(hero[1])
            if match_duration == 0.0: continue
            if safe_float(match.get("rank", 0)) <= 0: continue
            match['duration'] = match_duration
            for key in match.keys():
                if key == "player_id" or key == "py/object" or key == "rank": continue
                if key == "heroes_played":
                    continue
                match_stats[key] = safe_float(match[key])
            player_match_data.append(match_stats)
        all_players_data.extend(player_match_data)
    return pd.DataFrame(all_players_data)

def main():
    individual_data = load_individual_data('data/individual_player_stats.json')
    print(len(individual_data))
    """
    # predicting whether an individual player was on the winning or losing team based on their stats
    stats_df = build_dataframe(individual_data)
    print(stats_df.head(1))
    

    x = stats_df.drop(columns=['is_winner'])
    y = stats_df['is_winner']
    print(stats_df.shape)

    rbf_feature = RBFSampler(gamma='scale', n_components=100, random_state=42)
    x = rbf_feature.fit_transform(x)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    x_train = x[:round(.8*len(x))]
    x_test = x[round(.8*len(x)):]
    y_train = y[:round(.8*len(y))]     
    y_test = y[round(.8*len(y)):]
    
    
    wr_model = LinearSVC(random_state=13, max_iter=10000)
    wr_model.fit(x_train, y_train)
    #wr_model.fit(x[:round(.8*len(x))], y[:round(.8*len(y))])

    y_pred = wr_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy (WR): {accuracy}")
    """

    # predicting season-long W/L based on average stats per game

    stats_df = average_stats_per_player(individual_data)
    print(stats_df.head(1))
    print(stats_df.shape)

    x = stats_df.drop(columns=['rank','is_winner'])
    y = stats_df['is_winner']

    rbf_feature = RBFSampler(gamma='scale', n_components=48, random_state=42)
    x_features = rbf_feature.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=23)
    
    wr_model = LinearSVR(random_state=13, max_iter=10000)
    wr_model.fit(x_train, y_train)

    wr_baseline = [y_train.mean() for _ in range(len(y_test))]
    rmse_baseline = root_mean_squared_error(y_test, wr_baseline)
    mse_baseline = mean_squared_error(y_test, wr_baseline)
    y_pred = wr_model.predict(x_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE Baseline (WR): {round(mse_baseline,2)}")
    print(f"MSE (WR): {round(mse,2)}")
    print(f"RMSE Baseline (WR): {round(rmse_baseline,2)}")
    print(f"RMSE (WR): {round(rmse,2)}")

    # predicting a player's ELO based on average stats per game
    x = stats_df.drop(columns=['rank','is_winner'])
    y = stats_df['rank']

    rbf_feature = RBFSampler(gamma='scale', n_components=48, random_state=42)
    x_features = rbf_feature.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=23)
    
    elo_model = LinearSVR(random_state=13, max_iter=10000)
    elo_model.fit(x_train, y_train)
    
    elo_baseline = [y_train.mean() for _ in range(len(y_test))]
    rmse_baseline = root_mean_squared_error(y_test, elo_baseline)
    mse_baseline = mean_squared_error(y_test, elo_baseline)
    y_pred = elo_model.predict(x_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE Baseline (ELO): {round(mse_baseline,2)}")
    print(f"MSE (ELO): {round(mse,2)}")
    print(f"RMSE Baseline (ELO): {round(rmse_baseline,2)}")
    print(f"RMSE (ELO): {round(rmse,2)}")
    

if __name__ == "__main__":
    main()