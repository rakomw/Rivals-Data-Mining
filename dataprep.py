"""
Shared data preparation module for Marvel Rivals win prediction.
Reuses parsing patterns from svm.py for consistency.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Tuple

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
    """Load match data from JSON file."""
    with open(filepath, 'r') as fp:
        return json.load(fp)


def parse_number(value: str) -> float:
    """Parse numeric string with commas or percent to float."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(',', '').replace('%', ''))


def parse_time(time_str: str) -> int:
    """Parse time string to seconds."""
    if 's' in time_str:
        return round(float(time_str.replace("s", "")))
    else:
        return round(60 * float(time_str.replace("m", "")))


def get_match_duration(match: Dict[str, Any]) -> int:
    """Calculate match duration from player playtimes."""
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
    """Encode hero compositions as time-weighted vectors for each team."""
    valid_matches = []
    for m in match_data:
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

    return valid_matches


def extract_team_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract team-level statistics for each match."""
    team_stats = []

    for match in match_data:
        duration = get_match_duration(match)

        t1_kills = sum(parse_number(p['kills']) for p in match['team_one'])
        t1_deaths = sum(parse_number(p['deaths']) for p in match['team_one'])
        t1_assists = sum(parse_number(p['assists']) for p in match['team_one'])
        t1_final_hits = sum(parse_number(p['final_hits']) for p in match['team_one'])
        t1_solo_kills = sum(parse_number(p['solo_kills']) for p in match['team_one'])
        t1_damage = sum(parse_number(p['damage']) for p in match['team_one'])
        t1_damage_taken = sum(parse_number(p['damage_taken']) for p in match['team_one'])
        t1_damage_healed = sum(parse_number(p['damage_healed']) for p in match['team_one'])
        t1_accuracy = sum(parse_number(p['accuracy']) for p in match['team_one'])

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
    """Extract hero composition features from encoded match data."""
    hero_stats = []
    for match in match_data:
        hero_stats.append({
            'team_one_heroes': match['team_one_hero_vector'],
            'team_two_heroes': match['team_two_hero_vector']
        })

    df = pd.DataFrame(hero_stats)

    df = df.join(df.pop('team_one_heroes').apply(pd.Series).add_prefix('Team1_Hero_'))
    df = df.join(df.pop('team_two_heroes').apply(pd.Series).add_prefix('Team2_Hero_'))

    return df


def get_feature_sets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Get different feature configurations for comparison."""
    target_col = 'is_winner_team_one'

    basic_stat_keywords = ['kills', 'deaths', 'assists', 'damage', 'final_hits',
                           'solo_kills', 'accuracy', 'duration']
    stat_cols = [col for col in df.columns
                 if isinstance(col, str) and col != target_col
                 and any(x in col.lower() for x in basic_stat_keywords)]

    hero_cols = [col for col in df.columns
                 if isinstance(col, str) and col != target_col
                 and (col.startswith('Team1_Hero_') or col.startswith('Team2_Hero_'))]

    return {
        'team_stats': df[stat_cols] if stat_cols else df.drop(columns=[target_col], errors='ignore').iloc[:, :19],
        'hero_composition': df[hero_cols],
        'combined': df[stat_cols + hero_cols] if stat_cols else df[hero_cols],
    }


def get_train_test_split(X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_feature_info(X: pd.DataFrame) -> Dict[str, Any]:
    """Get feature set description."""
    stat_cols = [c for c in X.columns if not c.startswith('Team1_Hero_') and not c.startswith('Team2_Hero_')]
    hero_cols = [c for c in X.columns if c.startswith('Team1_Hero_') or c.startswith('Team2_Hero_')]

    return {
        'total_features': len(X.columns),
        'team_stats_features': len(stat_cols),
        'hero_composition_features': len(hero_cols),
        'num_heroes': len(HERO_MAP),
        'description': f"{len(X.columns)} features: {len(stat_cols)} team stats + {len(hero_cols)} hero composition ({len(HERO_MAP)} heroes x 2 teams)"
    }


def generate_csv(json_path: str = 'data/match_data.json',
                  csv_path: str = 'data/stats_and_heroes.csv') -> None:
    """Generate CSV from JSON data with random team swapping for class balance."""
    match_data = load_match_data(json_path)

    if isinstance(match_data, dict):
        match_data = list(match_data.values())

    hero_encoder = create_hero_encoder()
    match_data = encode_team_heroes(match_data, hero_encoder)

    hero_df = extract_hero_features(match_data)
    stats_df = extract_team_stats(match_data)

    full_df = stats_df.join(hero_df)

    # Randomly swap teams to balance classes (original data has all team_one wins)
    rng = np.random.RandomState(42)
    swap_indices = rng.rand(len(full_df)) < 0.5

    # Create column mapping for swaps (exclude is_winner_team_one)
    t1_stat_cols = [c for c in full_df.columns if 'team_one' in c and c != 'is_winner_team_one']
    t2_stat_cols = [c.replace('team_one', 'team_two') for c in t1_stat_cols]
    t1_hero_cols = [c for c in full_df.columns if c.startswith('Team1_Hero_')]
    t2_hero_cols = [c.replace('Team1_', 'Team2_') for c in t1_hero_cols]

    # Swap by creating new dataframe
    swapped_df = full_df.copy()
    for t1, t2 in zip(t1_stat_cols + t1_hero_cols, t2_stat_cols + t2_hero_cols):
        swapped_df.loc[swap_indices, t1] = full_df.loc[swap_indices, t2]
        swapped_df.loc[swap_indices, t2] = full_df.loc[swap_indices, t1]

    swapped_df.loc[swap_indices, 'is_winner_team_one'] = False

    swapped_df.to_csv(csv_path, index=False)


def load_data(csv_path: str = 'data/stats_and_heroes.csv',
              json_path: str = 'data/match_data.json') -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from CSV (generates from JSON if CSV doesn't exist)."""
    import os
    if not os.path.exists(csv_path):
        generate_csv(json_path, csv_path)

    df = pd.read_csv(csv_path)

    y = df['is_winner_team_one'].astype(int)
    X = df.drop(columns=['is_winner_team_one'])

    return X, y


if __name__ == '__main__':
    X, y = load_data()
