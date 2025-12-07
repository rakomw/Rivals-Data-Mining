import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
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

HERO_ROLE = {
    # Tanks
    "Emma Frost": "Tank",
    "Thor": "Tank",
    "Doctor Strange": "Tank",
    "Captain America": "Tank",
    "Angela": "Tank",
    "Hulk": "Tank",
    "Peni Parker": "Tank",
    "Venom": "Tank",
    "Groot": "Tank",
    "The Thing": "Tank",
    "Magneto": "Tank",

    # DPS
    "Magik": "DPS",
    "Storm": "DPS",
    "Mister Fantastic": "DPS",
    "Iron Fist": "DPS",
    "Black Panther": "DPS",
    "Hela": "DPS",
    "Iron Man": "DPS",
    "Star-Lord": "DPS",
    "Namor": "DPS",
    "Scarlet Witch": "DPS",
    "Human Torch": "DPS",
    "Spider-Man": "DPS",
    "Moon Knight": "DPS",
    "Winter Soldier": "DPS",
    "Psylocke": "DPS",
    "Phoenix": "DPS",
    "Blade": "DPS",
    "Hawkeye": "DPS",
    "Wolverine": "DPS",
    "The Punisher": "DPS",
    "Squirrel Girl": "DPS",
    "Black Widow": "DPS",
    "Daredevil": "DPS",

    # Support
    "Mantis": "Support",
    "Rocket Raccoon": "Support",
    "Ultron": "Support",
    "Jeff The Land Shark": "Support",
    "Adam Warlock": "Support",
    "Invisible Woman": "Support",
    "Cloak & Dagger": "Support",
    "Loki": "Support",
    "Luna Snow": "Support",
}

def main():
    individual_data = load_individual_data('data/individual_player_stats.json')
    print(len(individual_data))
    all_players_data = []
    for player_id in individual_data.keys():
        player_match_data = {}
        for match in individual_data[player_id]:
            """match_duration = 0.0
            for hero in match.get("heroes_played", []):
                match_duration += parse_minutes(hero[1])
            if match_duration == 0.0: continue"""
            player_match_data['time_played'] = 0.0
            for key in match.keys():
                if key == "player_id" or key == "py/object": continue
                if key == "heroes_played":
                    for hero in match[key]:
                        player_match_data['time_played'] += parse_minutes(hero[1])
                    continue
                if key not in player_match_data:
                    player_match_data[key] = 0
                player_match_data[key] += safe_float(match[key])
        for key in player_match_data.keys():
            player_match_data[key] /= len(individual_data[player_id])
            player_match_data[key] = round(player_match_data[key], 3)
        all_players_data.append(player_match_data)
    
    stats_df = pd.DataFrame(all_players_data)
    print(stats_df.head(1))
    print(stats_df.shape)

    x = stats_df.drop(columns=['rank','is_winner'])
    y = stats_df['is_winner']

    rbf_feature = RBFSampler(gamma='scale', n_components=48, random_state=42)
    x_features = rbf_feature.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=23)
    
    wr_model = LinearSVR(random_state=13, max_iter=1000)
    wr_model.fit(x_train, y_train)

    wr_baseline = [0.5 for _ in range(len(y_test))]
    mse_baseline = mean_squared_error(y_test, wr_baseline)
    print(f"MSE Baseline (WR): {mse_baseline}")

    y_pred = wr_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE (WR): {mse}")

    ranked_df = stats_df[stats_df['rank'] > 0]
    print("Ranked matches only: " + str(ranked_df.shape))
    x = ranked_df.drop(columns=['rank','is_winner'])
    y = ranked_df['is_winner']

    rbf_feature = RBFSampler(gamma='scale', n_components=48, random_state=42)
    x_features = rbf_feature.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=23)
    
    elo_model = LinearSVR(random_state=13, max_iter=1000)
    elo_model.fit(x_train, y_train)
    
    elo_baseline = [y_train.mean() for _ in range(len(y_test))]
    mse_baseline = mean_squared_error(y_test, elo_baseline)
    print(f"MSE Baseline (ELO): {mse_baseline}")
    y_pred = elo_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE (ELO): {mse}")

        

if __name__ == "__main__":
    main()