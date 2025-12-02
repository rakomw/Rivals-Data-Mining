# ============================================================
# nn_model.py
# Neural Network (NumPy) – Predict Match Winner (Team 1 vs Team 2)
# By Ken Douglas
#
# Goal:
#   Use cleaned match data (match_data_clean.json) to train a simple
#   neural network that predicts which team wins based on stats.
#
# Notes:
#   – This file DOES NOT clean data. The JSON file must already be cleaned.
#   – Output 1 ➜ Team One wins
#     Output 0 ➜ Team Two wins
# ============================================================

import os
import json
import numpy as np
from typing import List, Dict, Tuple

# ============================================================
# Load Cleaned Data
# ============================================================

def load_cleaned_matches(clean_path: str = "match_data_one_hero.json") -> List[Dict]:
    """Load pre-cleaned match JSON file."""
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"{clean_path} not found. Run cleaning.py first.")
    print(f"[INFO] Loading cleaned matches from {clean_path}")
    with open(clean_path, "r") as f:
        return json.load(f)

# ============================================================
# Utility Functions
# ============================================================

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


# Rank mapping (consistent with cleaning output)
RANK_ORDER = [
    "Unranked",
    "Bronze 3", "Bronze 2", "Bronze 1",
    "Silver 3", "Silver 2", "Silver 1",
    "Gold 3", "Gold 2", "Gold 1",
    "Platinum 3", "Platinum 2", "Platinum 1",
    "Diamond 3", "Diamond 2", "Diamond 1",
    "Grandmaster 3", "Grandmaster 2", "Grandmaster 1",
    "Celestial 3", "Celestial 2", "Celestial 1",
    "Eternity/One Above All",
]

def rank_score(rank: str) -> float:
    """Convert rank string → numeric value [0,1]."""
    if not isinstance(rank, str):
        return 0.0
    rank = rank.strip()
    if rank not in RANK_ORDER:
        return 0.0
    idx = RANK_ORDER.index(rank)
    return idx / (len(RANK_ORDER) - 1)

# Global map-name → numeric index mapping, built on the fly
MAP_INDEX: Dict[str, int] = {}

def map_to_numeric(map_name) -> float:
    """Convert a map string to a stable numeric id in [1, N].

    Unknown or missing maps map to 0. This is a simple categorical
    encoding; the actual ids are assigned in the order maps are seen.
    """
    if not isinstance(map_name, str):
        return 0.0
    name = map_name.strip()
    if not name:
        return 0.0
    if name not in MAP_INDEX:
        MAP_INDEX[name] = len(MAP_INDEX) + 1  # start ids at 1
    return float(MAP_INDEX[name])

# ============================================================
# Hero Role Mapping (hero-name → role)
# ============================================================

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

# ============================================================
# Team-Up Mapping (anchor hero -> list of recipient heroes)
# ============================================================
# This allows us to track special team-up synergies, where an
# "anchor" hero has one or more teammates that enable a combo
# (e.g., team-up ultimates, buff synergies, etc.).
TEAM_UPS: Dict[str, List[str]] = {
    "Emma Frost": ["Psylocke", "Magneto"],
    "Doctor Strange": ["Scarlet Witch", "Magik"],
    "Cloak & Dagger": ["Blade", "Moon Knight"],
    "The Punisher": ["Black Widow"],
    "Rocket Raccoon": ["Peni Parker", "Star-Lord"],
    "Mantis": ["Groot", "Loki"],
    "Hela": ["Namor"],
    "Venom": ["Hela", "Jeff The Land Shark"],
    "Invisible Woman": ["Human Torch", "Mister Fantastic", "The Thing"],
    "Luna Snow": ["Hawkeye", "Iron Fist"],
    "Daredevil": ["The Punisher"],
    "Winter Soldier": ["Captain America"],
    "Iron Man": ["Ultron", "Squirrel Girl"],
    "Phoenix": ["Wolverine"],
    "Human Torch": ["Spider-Man", "The Thing"],
    "Adam Warlock": ["Luna Snow"],
    "Angela": ["Thor"],
    "Groot": ["Jeff The Land Shark", "Rocket Raccoon"],
    "Wolverine": ["The Thing", "Hulk"],
    "Hulk": ["Black Panther"],
}


# ============================================================
# Feature Engineering
# ============================================================

# change this to account for hero stats similar to the svm method from Robert's implementation
def team_features(players: List[Dict]) -> np.ndarray:
    """Aggregate stats for a single team."""
    dmg = 0.0
    taken = 0.0
    heal = 0.0
    minutes = 0.0
    deaths = 0.0
    hero_count = 0.0
    ranks = []
    #accuracy = 0.0
    #solo = 0.0
    kills = 0.0
    assist = 0.0
    finals = 0.0

    # Team-level role minutes
    tank_minutes_team = 0.0
    dps_minutes_team = 0.0
    support_minutes_team = 0.0

    # Set of all heroes that appeared on this team (for team-up checks)
    team_heroes = set()

    # Team-up tracking: how many anchor+recipient combos are present
    team_teamup_count = 0.0
    team_has_any_teamup = 0.0  # 0.0 or 1.0 as a feature flag

    # Per-player tracking (for team-level aggregates later)
    player_damage_list = []
    player_deaths_list = []
    player_carry_score_list = []  # deaths-aware carry score per player

    player_tank_carry_list = []
    player_dps_carry_list = []
    player_support_carry_list = []

    # Per-player per-minute tracking
    player_dpm_list = []
    player_hpm_list = []
    player_tpm_list = []

    # Swap and hero-consistency tracking
    team_swap_count = 0.0
    player_hero_count_list = []
    player_main_hero_share_list = []

    for p in players or []:
        # Raw player totals
        player_damage = safe_float(p.get("damage"))
        player_taken = safe_float(p.get("damage_taken"))
        player_heal = safe_float(p.get("damage_healed"))
        player_deaths = safe_float(p.get("deaths"))
        player_kills = safe_float(p.get("kills"))
        player_assists = safe_float(p.get("assists"))
        player_finals = safe_float(p.get("final_hits"))

        dmg += player_damage
        taken += player_taken
        heal += player_heal
        deaths += player_deaths

        # Track individual damage and deaths for carry-vs-team features
        player_damage_list.append(player_damage)
        player_deaths_list.append(player_deaths)

        ranks.append(rank_score(p.get("rank")))
        #accuracy += safe_float(p.get("accuracy"))
        #solo += safe_float(p.get("solo_kills"))
        kills += player_kills
        assist += player_assists
        finals += player_finals

        # Per-player minutes from heroes
        player_minutes = 0.0
        player_role_minutes = {"Tank": 0.0, "DPS": 0.0, "Support": 0.0}
        hero_minutes_map = {}
        heroes = p.get("heroes_played", [])
        for entry in heroes:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                hero_name = str(entry[0])
                role = HERO_ROLE.get(hero_name)
                m = parse_minutes(entry[1])

                # Track that this hero appeared on this team at all
                team_heroes.add(hero_name)

                player_minutes += m
                minutes += m
                hero_count += 1.0

                # Track hero-specific minutes for this player
                hero_minutes_map[hero_name] = hero_minutes_map.get(hero_name, 0.0) + m

                if role == "Tank":
                    player_role_minutes["Tank"] += m
                    tank_minutes_team += m
                elif role == "DPS":
                    player_role_minutes["DPS"] += m
                    dps_minutes_team += m
                elif role == "Support":
                    player_role_minutes["Support"] += m
                    support_minutes_team += m

        # Hero swap and consistency stats per player
        hero_count_player = len(hero_minutes_map)
        player_hero_count_list.append(hero_count_player)
        if hero_count_player > 0 and player_minutes > 0:
            main_hero_minutes = max(hero_minutes_map.values())
            main_hero_share = main_hero_minutes / player_minutes
        else:
            main_hero_share = 0.0
        player_main_hero_share_list.append(main_hero_share)

        # Swaps per player: playing N heroes counts as (N-1) swaps (min 0)
        if hero_count_player > 1:
            team_swap_count += (hero_count_player - 1)

        # Per-player per-minute stats, if they actually played
        if player_minutes > 0:
            player_dpm_list.append(player_damage / player_minutes)
            player_hpm_list.append(player_heal / player_minutes)
            player_tpm_list.append(player_taken / player_minutes)

        # Deaths-aware carry score: combines damage, healing, damage_taken, and final hits,
        # rewarding high impact with low deaths. Tanks (high taken, low deaths), supports
        # (high healing + clutch finals), and DPS (high damage + finals) can all surface
        # as "carry" in different ways.
        carry_score = (player_damage + player_heal + player_taken + player_finals) / (player_deaths + 1.0)
        player_carry_score_list.append(carry_score)

        # Assign this carry score to the player's primary role (based on minutes played)
        primary_role = max(player_role_minutes, key=player_role_minutes.get)
        if player_role_minutes[primary_role] > 0:
            if primary_role == "Tank":
                player_tank_carry_list.append(carry_score)
            elif primary_role == "DPS":
                player_dps_carry_list.append(carry_score)
            elif primary_role == "Support":
                player_support_carry_list.append(carry_score)

    # Team-up feature computation: check each anchor hero and its
    # recipients to see if they co-occurred on this team.
    if TEAM_UPS and team_heroes:
        for anchor, recipients in TEAM_UPS.items():
            if anchor in team_heroes:
                for r in recipients:
                    if r in team_heroes:
                        team_teamup_count += 1.0
        if team_teamup_count > 0:
            team_has_any_teamup = 1.0

    avg_rank = np.mean(ranks) if ranks else 0.0
    avg_hero_minutes = minutes / hero_count if hero_count > 0 else 0.0

    # Team-level per-minute stats (using total team damage/time)
    if minutes > 0:
        dmg_per_min_team = dmg / minutes
        heal_per_min_team = heal / minutes
        taken_per_min_team = taken / minutes
    else:
        dmg_per_min_team = 0.0
        heal_per_min_team = 0.0
        taken_per_min_team = 0.0

    # Aggregated per-player per-minute stats
    mean_player_dpm = float(np.mean(player_dpm_list)) if player_dpm_list else 0.0
    mean_player_hpm = float(np.mean(player_hpm_list)) if player_hpm_list else 0.0
    mean_player_tpm = float(np.mean(player_tpm_list)) if player_tpm_list else 0.0

    # Aggregated deaths stats (team and per-player)
    if player_deaths_list:
        deaths_arr = np.array(player_deaths_list, dtype=np.float64)
        mean_player_deaths = float(deaths_arr.mean())
        max_player_deaths = float(deaths_arr.max())
        std_player_deaths = float(deaths_arr.std())
    else:
        mean_player_deaths = 0.0
        max_player_deaths = 0.0
        std_player_deaths = 0.0

    # Deaths-aware carry scores
    if player_carry_score_list:
        carry_arr = np.array(player_carry_score_list, dtype=np.float64)
        max_carry_score = float(carry_arr.max())
        mean_carry_score = float(carry_arr.mean())
        std_carry_score = float(carry_arr.std())
    else:
        max_carry_score = 0.0
        mean_carry_score = 0.0
        std_carry_score = 0.0

    # Carry vs. team damage features
    if player_damage_list:
        damage_arr = np.array(player_damage_list, dtype=np.float64)
        max_player_damage = float(damage_arr.max())
        mean_player_damage = float(damage_arr.mean())
        std_player_damage = float(damage_arr.std())
        carry_share = max_player_damage / (dmg + 1e-6) if dmg > 0 else 0.0
    else:
        max_player_damage = 0.0
        mean_player_damage = 0.0
        std_player_damage = 0.0
        carry_share = 0.0

    # Per-player DPM distribution (carry vs team tempo)
    if player_dpm_list:
        dpm_arr = np.array(player_dpm_list, dtype=np.float64)
        max_player_dpm = float(dpm_arr.max())
        std_player_dpm = float(dpm_arr.std())
    else:
        max_player_dpm = 0.0
        std_player_dpm = 0.0

    # Hero swap / consistency aggregates
    if player_hero_count_list:
        hero_count_arr = np.array(player_hero_count_list, dtype=np.float64)
        mean_heroes_per_player = float(hero_count_arr.mean())
        max_heroes_per_player = float(hero_count_arr.max())
        std_heroes_per_player = float(hero_count_arr.std())
    else:
        mean_heroes_per_player = 0.0
        max_heroes_per_player = 0.0
        std_heroes_per_player = 0.0

    if player_main_hero_share_list:
        main_share_arr = np.array(player_main_hero_share_list, dtype=np.float64)
        mean_main_hero_share = float(main_share_arr.mean())
        max_main_hero_share = float(main_share_arr.max())
        std_main_hero_share = float(main_share_arr.std())
    else:
        mean_main_hero_share = 0.0
        max_main_hero_share = 0.0
        std_main_hero_share = 0.0

    # Role composition: minutes and shares
    total_role_minutes = tank_minutes_team + dps_minutes_team + support_minutes_team
    if total_role_minutes > 0:
        tank_share = tank_minutes_team / total_role_minutes
        dps_share = dps_minutes_team / total_role_minutes
        support_share = support_minutes_team / total_role_minutes
    else:
        tank_share = dps_share = support_share = 0.0

    # Role-based carry scores (best per role)
    max_tank_carry = float(max(player_tank_carry_list)) if player_tank_carry_list else 0.0
    max_dps_carry = float(max(player_dps_carry_list)) if player_dps_carry_list else 0.0
    max_support_carry = float(max(player_support_carry_list)) if player_support_carry_list else 0.0

    return np.array([
        dmg,
        taken,
        heal,
        minutes,
        deaths,
        avg_rank,
        kills,
        assist,
        finals,
        hero_count,
        avg_hero_minutes,
        dmg_per_min_team,
        heal_per_min_team,
        taken_per_min_team,
        mean_player_dpm,
        mean_player_hpm,
        mean_player_tpm,
        mean_player_deaths,
        max_player_deaths,
        std_player_deaths,
        max_player_damage,
        mean_player_damage,
        std_player_damage,
        carry_share,
        max_player_dpm,
        std_player_dpm,
        max_carry_score,
        mean_carry_score,
        std_carry_score,
        # Swap / hero consistency features
        team_swap_count,
        mean_heroes_per_player,
        max_heroes_per_player,
        std_heroes_per_player,
        mean_main_hero_share,
        max_main_hero_share,
        std_main_hero_share,
        # Role composition
        tank_minutes_team,
        dps_minutes_team,
        support_minutes_team,
        tank_share,
        dps_share,
        support_share,
        # Role-based carry highlights
        max_tank_carry,
        max_dps_carry,
        max_support_carry,
        # Team-up features
        team_teamup_count,
        team_has_any_teamup,

    ], dtype=np.float64)  # hero-based, deaths-based, per-player-minute, swap/consistency, role-composition, role-based carry, and team-up features


def build_feature_vector(match: Dict) -> np.ndarray:
    """Create full input vector from team stats plus match-level context.

    We return [team1_features, team2_features, map_id].

    NOTE:
        Final team scores are intentionally *not* included as input features,
        because they are strongly tied to the match outcome (winner label) and
        would cause target leakage. Scores should instead be treated like the
        winner label: something we reveal/evaluate at the end of the match,
        not as a predictor.
    """
    t1 = team_features(match.get("team_one"))
    t2 = team_features(match.get("team_two"))

    # --- Match-level context (map only) ---
    # Map name → numeric id. This is safe to use as an input feature because
    # the map is known before or at match start, unlike the final score.
    map_name = (
        match.get("map")
        or match.get("map_name")
        or match.get("mapName")
        or ""
    )
    map_id = map_to_numeric(map_name)

    # NOTE: Scores are not included in the input features to avoid leakage.
    # Scores are revealed only as part of labels/analysis, not as model inputs.
    extra = np.array(
        [
            map_id,
        ],
        dtype=np.float32,
    )

    # Current implementation: use team1, team2, and match-level extras (map).
    full_vec = np.concatenate([t1, t2, extra], axis=0)
    return full_vec

    # Experimental implementation (diff-only), kept for reference:
    # diff = t1 - t2
    # return diff


def prepare_dataset(matches: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert match list → feature matrix X and labels y."""
    X, y = [], []
    skipped = 0

    for m in matches:
        w = m.get("winner")
        if w not in (1, 2):
            skipped += 1
            continue

        X.append(build_feature_vector(m))
        y.append([1 if w == 1 else 0])

    if skipped:
        print(f"[INFO] Skipped {skipped} matches (missing or invalid winner).")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ============================================================
# Train/Test Split
# ============================================================

def split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    return X[:cut], X[cut:], y[:cut], y[cut:]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

# ============================================================
# Neural Network (NumPy)
# ============================================================

class NeuralNetwork:
    """Two-hidden-layer neural network (ReLU hidden, sigmoid output)."""
    def __init__(self, input_dim, hidden_dim=32, hidden_dim2=16, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)

        # Layer 1: input -> hidden_dim
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        # Layer 2: hidden_dim -> hidden_dim2
        self.W2 = rng.normal(0, 0.1, (hidden_dim, hidden_dim2)).astype(np.float32)
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)

        # Output layer: hidden_dim2 -> 1
        self.W3 = rng.normal(0, 0.1, (hidden_dim2, 1)).astype(np.float32)
        self.b3 = np.zeros((1, 1), dtype=np.float32)

        self.lr = lr

    # ----- Activations -----
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        # Leaky ReLU: small negative slope to prevent dead neurons
        return np.where(z > 0, z, 0.01 * z)

    def relu_deriv(self, z):
        # Derivative of Leaky ReLU
        grad = np.ones_like(z)
        grad[z < 0] = 0.01
        return grad

    # ----- Forward -----
    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self.relu(Z2)
        Z3 = A2 @ self.W3 + self.b3
        A3 = self.sigmoid(Z3)  # sigmoid only on output for binary classification
        return A3, {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "A3": A3}

    # ----- Loss -----
    def loss(self, y, y_pred):
        """Binary cross-entropy loss with defensive clipping."""
        eps = 1e-7
        # Ensure array and dtype
        y_pred = np.asarray(y_pred, dtype=np.float32)
        # Replace non-finite predictions with 0.5 (uninformative)
        bad_mask = ~np.isfinite(y_pred)
        if np.any(bad_mask):
            y_pred[bad_mask] = 0.5
        # Clip away from 0 and 1 to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # ----- Backprop -----
    def backward(self, y, cache):
        X = cache["X"]
        Z1, A1 = cache["Z1"], cache["A1"]
        Z2, A2 = cache["Z2"], cache["A2"]
        A3 = cache["A3"]
        m = len(X)

        # Output layer (sigmoid + BCE loss)
        dZ3 = A3 - y
        dW3 = (A2.T @ dZ3) / m
        db3 = np.mean(dZ3, axis=0, keepdims=True)

        # Hidden layer 2 (ReLU)
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_deriv(Z2)
        dW2 = (A1.T @ dZ2) / m
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        # Hidden layer 1 (ReLU)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_deriv(Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # Gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    # ----- Train -----
    def fit(self, X, y, epochs=2000, print_every=200, batch_size=1024,
            X_val=None, y_val=None, patience=10):
        """
        Train the network using mini-batch gradient descent.

        Args:
            X, y: training data
            epochs: maximum number of epochs
            print_every: logging frequency
            batch_size: mini-batch size
            X_val, y_val: optional validation set for early stopping
            patience: number of epochs without validation loss improvement
                      before stopping early
        """
        m = len(X)
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # Shuffle training data indices each epoch
            idx = np.random.permutation(m)

            # Mini-batch loop
            for start in range(0, m, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                y_pred, cache = self.forward(X_batch)
                self.backward(y_batch, cache)

            # Periodic logging + validation
            if epoch % print_every == 0 or epoch == 1:
                # Compute training loss on full training data
                train_pred, _ = self.forward(X)
                train_loss = self.loss(y, train_pred)

                msg = f"[EPOCH {epoch}] Train Loss: {train_loss:.4f}"

                if X_val is not None and y_val is not None:
                    val_pred, _ = self.forward(X_val)
                    val_loss = self.loss(y_val, val_pred)
                    msg += f" | Val Loss: {val_loss:.4f}"

                    # Early stopping check
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        print(msg + " | Early stopping triggered.")
                        break

                print(msg)

            # If early stopping triggered, break outer loop as well
            if epochs_no_improve >= patience:
                break

    # ----- Predict -----
    def predict(self, X):
        y_pred, _ = self.forward(X)
        return (y_pred >= 0.5).astype(int)


# ============================================================
# Main Training Routine
# ============================================================

def main():
    matches = load_cleaned_matches()
    X, y = prepare_dataset(matches)

    # First split: train+val vs test
    X_train_full, X_test, y_train_full, y_test = split(X, y, test_size=0.2)

    # Second split: train vs val (from the training portion)
    X_train, X_val, y_train, y_val = split(X_train_full, y_train_full, test_size=0.1)

    # Standardize using only training data statistics
    X_train, X_val, mean, std = standardize(X_train, X_val)
    X_test = (X_test - mean) / std

    model = NeuralNetwork(input_dim=X_train.shape[1], hidden_dim=16, lr=0.001)

    print("[INFO] Training neural network with mini-batches and early stopping...")
    model.fit(
        X_train,
        y_train,
        epochs=4000,
        print_every=200,
        batch_size=1024,
        X_val=X_val,
        y_val=y_val,
        patience=10,
    )

    train_acc = np.mean(model.predict(X_train) == y_train)
    val_acc = np.mean(model.predict(X_val) == y_val)
    test_acc = np.mean(model.predict(X_test) == y_test)

    print(f"[RESULT] Train Accuracy: {train_acc:.3f}")
    print(f"[RESULT] Val Accuracy  : {val_acc:.3f}")
    print(f"[RESULT] Test Accuracy : {test_acc:.3f}")

    np.savez(
        "nn_weights_and_stats.npz",
        W1=model.W1, b1=model.b1,
        W2=model.W2, b2=model.b2,
        W3=model.W3, b3=model.b3,
        mean=mean, std=std
    )
    print("[INFO] Saved model parameters → nn_weights_and_stats.npz")


# ============================================================

if __name__ == "__main__":
    main()