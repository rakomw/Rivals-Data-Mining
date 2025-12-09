"""
Shared data preparation module for Marvel Rivals win prediction.
Reuses parsing patterns from svm.py for consistency.
Uses nn_model for advanced feature extraction.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple
import nn_model

# Hero ID to name mapping (43 heroes across 3 roles)
HERO_MAP = {
    # Tanks (11)
    "1053001": "Emma Frost",
    "1039001": "Thor",
    "1018001": "Doctor Strange",
    "1022001": "Captain America",
    "1056001": "Angela",
    "1011001": "Hulk",
    "1042001": "Peni Parker",
    "1035001": "Venom",
    "1027001": "Groot",
    "1051001": "The Thing",
    "1037001": "Magneto",

    # DPS (23)
    "1029001": "Magik",
    "1015001": "Storm",
    "1040001": "Mister Fantastic",
    "1052001": "Iron Fist",
    "1026001": "Black Panther",
    "1024001": "Hela",
    "1034001": "Iron Man",
    "1043001": "Star-Lord",
    "1045001": "Namor",
    "1038001": "Scarlet Witch",
    "1017001": "Human Torch",
    "1036001": "Spider-Man",
    "1030001": "Moon Knight",
    "1041001": "Winter Soldier",
    "1048001": "Psylocke",
    "1054001": "Phoenix",
    "1044001": "Blade",
    "1021001": "Hawkeye",
    "1049001": "Wolverine",
    "1014001": "The Punisher",
    "1032001": "Squirrel Girl",
    "1033001": "Black Widow",
    "1055001": "Daredevil",

    # Support (9)
    "1020001": "Mantis",
    "1023001": "Rocket Raccoon",
    "1028001": "Ultron",
    "1047001": "Jeff The Land Shark",
    "1046001": "Adam Warlock",
    "1050001": "Invisible Woman",
    "1025001": "Cloak & Dagger",
    "1016001": "Loki",
    "1031001": "Luna Snow"
}

# Hero role mapping
HERO_ROLES = {
    # Tanks
    "1053001": "Tank", "1039001": "Tank", "1018001": "Tank", "1022001": "Tank",
    "1056001": "Tank", "1011001": "Tank", "1042001": "Tank", "1035001": "Tank",
    "1027001": "Tank", "1051001": "Tank", "1037001": "Tank",
    # DPS
    "1029001": "DPS", "1015001": "DPS", "1040001": "DPS", "1052001": "DPS",
    "1026001": "DPS", "1024001": "DPS", "1034001": "DPS", "1043001": "DPS",
    "1045001": "DPS", "1038001": "DPS", "1017001": "DPS", "1036001": "DPS",
    "1030001": "DPS", "1041001": "DPS", "1048001": "DPS", "1054001": "DPS",
    "1044001": "DPS", "1021001": "DPS", "1049001": "DPS", "1014001": "DPS",
    "1032001": "DPS", "1033001": "DPS", "1055001": "DPS",
    # Support
    "1020001": "Support", "1023001": "Support", "1028001": "Support",
    "1047001": "Support", "1046001": "Support", "1050001": "Support",
    "1025001": "Support", "1016001": "Support", "1031001": "Support"
}


def load_match_data(filepath: str) -> List[Dict[str, Any]]:
    """Load match data from JSON file.

    Args:
        filepath: Path to the match_data.json file.

    Returns:
        List of match dictionaries containing player and game information.
    """
    with open(filepath, 'r') as fp:
        return json.load(fp)


def parse_number(value: str) -> float:
    """Parse numeric string with commas or percent to float.

    Args:
        value: Numeric string potentially containing commas (e.g., "1,234") or percent (e.g., "50%").

    Returns:
        Float representation of the number.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(',', '').replace('%', ''))


def parse_time(time_str: str) -> int:
    """Parse time string to seconds.

    Args:
        time_str: Time string in minutes or seconds format (e.g., "35.00m" or "19s").

    Returns:
        Total time in seconds.
    """
    if 's' in time_str:
        # Seconds format
        return round(float(time_str.replace("s", "")))
    else:
        # Minutes format
        return round(60 * float(time_str.replace("m", "")))


def get_match_duration(match: Dict[str, Any]) -> int:
    """Calculate match duration from player playtimes.

    Args:
        match: Match dictionary containing team_one and team_two.

    Returns:
        Match duration in seconds (longest playtime across all players).
    """
    max_duration = 0

    for team in [match['team_one'], match['team_two']]:
        for player in team:
            player_duration = 0
            for _, playtime in player['heroes_played']:
                player_duration += parse_time(playtime)
            max_duration = max(max_duration, player_duration)

    return max_duration


def create_hero_encoder() -> Dict[str, int]:
    """Create a mapping from hero name to index for encoding."""
    return {hero_name: idx for idx, hero_name in enumerate(HERO_MAP.values())}


def encode_team_heroes(match_data: List[Dict[str, Any]], hero_encoder: Dict[str, int]) -> List[Dict[str, Any]]:
    """Encode hero compositions as time-weighted vectors for each team.

    Args:
        match_data: List of match dictionaries.
        hero_encoder: Mapping from hero ID to index.

    Returns:
        Match data with added team_one_hero_vector and team_two_hero_vector.
    """
    valid_matches = []
    for m in match_data:
        # Skip invalid matches
        if m.get('team_one') is None or m.get('team_two') is None or m.get('winner') is None:
            continue

        for team in ['team_one', 'team_two']:
            hero_vector = [0.0] * len(hero_encoder)
            for player in m[team]:
                total_time = sum(parse_time(hero[1]) for hero in player["heroes_played"])
                if total_time > 0:
                    for hero_name, playtime in player["heroes_played"]:
                        if hero_name in hero_encoder:
                            hero_vector[hero_encoder[hero_name]] += parse_time(playtime) / total_time
            m[team + "_hero_vector"] = hero_vector
        valid_matches.append(m)

    print(f"Filtered to {len(valid_matches)} valid matches")
    return valid_matches


def extract_team_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract team-level statistics for each match.

    Args:
        match_data: List of match dictionaries from the JSON file.

    Returns:
        DataFrame with team statistics and match outcomes.
    """
    team_stats = []

    for match in match_data:
        duration = get_match_duration(match)

        # Team one stats
        t1_kills = sum(parse_number(p['kills']) for p in match['team_one'])
        t1_deaths = sum(parse_number(p['deaths']) for p in match['team_one'])
        t1_assists = sum(parse_number(p['assists']) for p in match['team_one'])
        t1_final_hits = sum(parse_number(p['final_hits']) for p in match['team_one'])
        t1_solo_kills = sum(parse_number(p['solo_kills']) for p in match['team_one'])
        t1_damage = sum(parse_number(p['damage']) for p in match['team_one'])
        t1_damage_taken = sum(parse_number(p['damage_taken']) for p in match['team_one'])
        t1_damage_healed = sum(parse_number(p['damage_healed']) for p in match['team_one'])
        t1_accuracy = sum(parse_number(p['accuracy']) for p in match['team_one'])

        # Team two stats
        t2_kills = sum(parse_number(p['kills']) for p in match['team_two'])
        t2_deaths = sum(parse_number(p['deaths']) for p in match['team_two'])
        t2_assists = sum(parse_number(p['assists']) for p in match['team_two'])
        t2_final_hits = sum(parse_number(p['final_hits']) for p in match['team_two'])
        t2_solo_kills = sum(parse_number(p['solo_kills']) for p in match['team_two'])
        t2_damage = sum(parse_number(p['damage']) for p in match['team_two'])
        t2_damage_taken = sum(parse_number(p['damage_taken']) for p in match['team_two'])
        t2_damage_healed = sum(parse_number(p['damage_healed']) for p in match['team_two'])
        t2_accuracy = sum(parse_number(p['accuracy']) for p in match['team_two'])

        is_winner_team_one = (match['winner'] == 1)

        team_stats.append({
            'match_duration': duration,
            'team_one_kills': t1_kills,
            'team_one_deaths': t1_deaths,
            'team_one_assists': t1_assists,
            'team_one_final_hits': t1_final_hits,
            'team_one_solo_kills': t1_solo_kills,
            'team_one_damage': t1_damage,
            'team_one_damage_taken': t1_damage_taken,
            'team_one_damage_healed': t1_damage_healed,
            'team_one_accuracy': t1_accuracy,
            'team_two_kills': t2_kills,
            'team_two_deaths': t2_deaths,
            'team_two_assists': t2_assists,
            'team_two_final_hits': t2_final_hits,
            'team_two_solo_kills': t2_solo_kills,
            'team_two_damage': t2_damage,
            'team_two_damage_taken': t2_damage_taken,
            'team_two_damage_healed': t2_damage_healed,
            'team_two_accuracy': t2_accuracy,
            'is_winner_team_one': is_winner_team_one
        })

    return pd.DataFrame(team_stats)


def extract_hero_features(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract hero composition features from encoded match data.

    Args:
        match_data: List of match dictionaries with hero vectors.

    Returns:
        DataFrame with hero vectors for both teams.
    """
    hero_stats = []
    for match in match_data:
        hero_stats.append({
            'team_one_heroes': match['team_one_hero_vector'],
            'team_two_heroes': match['team_two_hero_vector']
        })

    df = pd.DataFrame(hero_stats)

    # Expand hero vectors into columns (matching svm.py format)
    df = df.join(df.pop('team_one_heroes').apply(pd.Series).add_prefix('Team1_Hero_'))
    df = df.join(df.pop('team_two_heroes').apply(pd.Series).add_prefix('Team2_Hero_'))

    return df


def get_feature_sets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Get different feature configurations for comparison.

    Args:
        df: Full DataFrame with all features (full features + hero vectors).

    Returns:
        Dictionary with feature set names and corresponding DataFrames.
        - team_stats: Basic stats columns only (~19 features)
        - hero_composition: Hero vectors only (86 features)
        - combined: team_stats + hero_composition (~105 features)
        - full: All features including nn_model advanced features (~181 features)
    """
    # Basic stats (subset that matches original team_stats)
    basic_stat_keywords = ['kills', 'deaths', 'assists', 'damage', 'final_hits',
                           'solo_kills', 'accuracy', 'duration']
    stat_cols = [col for col in df.columns if any(x in col.lower() for x in basic_stat_keywords)]

    # Hero vectors (86 features)
    hero_cols = [col for col in df.columns if col.startswith('Team1_') or col.startswith('Team2_')]

    return {
        'team_stats': df[stat_cols] if stat_cols else df.iloc[:, :19],
        'hero_composition': df[hero_cols],
        'combined': df[stat_cols + hero_cols] if stat_cols else df[hero_cols],
        'full': df
    }


def get_train_test_split(X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_data(json_path: str = 'data/match_data_one_hero.json') -> Tuple[pd.DataFrame, pd.Series]:
    """Load data directly from cleaned JSON file using nn_model for advanced features.

    Args:
        json_path: Path to the cleaned JSON file.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    print(f"Loading from {json_path}...")
    match_data = load_match_data(json_path)

    # Convert dict to list if needed
    if isinstance(match_data, dict):
        match_data = list(match_data.values())

    print(f"Loaded {len(match_data)} matches")

    # Encode hero compositions
    hero_encoder = create_hero_encoder()
    match_data = encode_team_heroes(match_data, hero_encoder)

    # Extract hero features (86 columns)
    hero_df = extract_hero_features(match_data)

    # Use nn_model for advanced team features (same as svm.py)
    print("Extracting advanced features via nn_model...")
    team_stats, winners = nn_model.prepare_dataset(match_data)
    stats_df = pd.DataFrame(team_stats)

    # Combine: nn_model features + hero vectors
    full_df = stats_df.join(hero_df)
    full_df['is_winner_team_one'] = winners.flatten().astype(int)

    y = full_df['is_winner_team_one']
    X = full_df.drop(columns=['is_winner_team_one'])

    print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    return X, y


if __name__ == '__main__':
    # Test loading from JSON
    print("Testing data loading...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"\nFeature columns:\n{X.columns.tolist()[:10]}...")
