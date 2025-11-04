import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import List, Dict, Any
import re

def load_match_data(filepath: str) -> List[Dict[str, Any]]:
    """Load match data from JSON file.

    Args:
        filepath: Path to the match_data.json file.

    Returns:
        List of match dictionaries containing player and game information.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(filepath, 'r') as fp:
        return json.load(fp)
    
def extract_match_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract team-level statistics for each match.

    Args:
        match_data: List of match dictionaries from the JSON file.

    Returns:
        DataFrame with columns containing team statistics and match outcomes.
    """
    team_stats = []

    for match in match_data:
        duration = get_match_duration(match)

        team_one_kills = sum(parse_number(p['kills']) for p in match['team_one'])
        team_one_deaths = sum(parse_number(p['deaths']) for p in match['team_one'])
        team_one_assists = sum(parse_number(p['assists']) for p in match['team_one'])
        team_one_final_hits = sum(parse_number(p['final_hits']) for p in match['team_one'])
        team_one_solo_kills = sum(parse_number(p['solo_kills']) for p in match['team_one'])
        team_one_damage = sum(parse_number(p['damage']) for p in match['team_one'])
        team_one_damage_taken = sum(parse_number(p['damage_taken']) for p in match['team_one'])
        team_one_damage_healed = sum(parse_number(p['damage_healed']) for p in match['team_one'])
        team_one_accuracy = sum(parse_number(p['accuracy']) for p in match['team_one'])

        team_two_kills = sum(parse_number(p['kills']) for p in match['team_two'])
        team_two_deaths = sum(parse_number(p['deaths']) for p in match['team_two'])
        team_two_assists = sum(parse_number(p['assists']) for p in match['team_two'])
        team_two_final_hits = sum(parse_number(p['final_hits']) for p in match['team_two'])
        team_two_solo_kills = sum(parse_number(p['solo_kills']) for p in match['team_two'])
        team_two_damage = sum(parse_number(p['damage']) for p in match['team_two'])
        team_two_damage_taken = sum(parse_number(p['damage_taken']) for p in match['team_two'])
        team_two_damage_healed = sum(parse_number(p['damage_healed']) for p in match['team_two'])
        team_two_accuracy = sum(parse_number(p['accuracy']) for p in match['team_two'])

        is_winner_team_one = (match['winner'] == 1)

        team_stats.append({
            'match_duration': duration,
            'team_one_kills': team_one_kills,
            'team_one_deaths': team_one_deaths,
            'team_one_assists': team_one_assists,
            'team_one_final_hits': team_one_final_hits,
            'team_one_solo_kills': team_one_solo_kills,
            'team_one_damage': team_one_damage,
            'team_one_damage_taken': team_one_damage_taken,
            'team_one_damage_healed': team_one_damage_healed,
            'team_one_accuracy': team_one_accuracy,
            'team_two_kills': team_two_kills,
            'team_two_deaths': team_two_deaths,
            'team_two_assists': team_two_assists,
            'team_two_final_hits': team_two_final_hits,
            'team_two_solo_kills': team_two_solo_kills,
            'team_two_damage': team_two_damage,
            'team_two_damage_taken': team_two_damage_taken,
            'team_two_damage_healed': team_two_damage_healed,
            'team_two_accuracy': team_two_accuracy,
            'is_winner_team_one': is_winner_team_one
        })

    return pd.DataFrame(team_stats)

def parse_number(value: str) -> float:
    """Parse numeric string with commas to float.

    Args:
        value: Numeric string potentially containing commas (e.g., "1,234").

    Returns:
        Float representation of the number.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(',', ''))

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

def parse_time(time_str: str) -> int:
    """Parse time string to seconds.

    Args:
        time_str: Time string in minutes format (e.g., "35.00m").

    Returns:
        Total time in seconds.
    """
    time_str = time_str.replace("m","")
    return round(60 * float(time_str))

def main():
    match_data = load_match_data('data/match_data_clean.json')
    stats_df = extract_match_stats(match_data)

    print(stats_df.head(5))

    x = stats_df.drop(columns=['is_winner_team_one'])
    y = stats_df['is_winner_team_one'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
    model = SVC(random_state=13)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    main()