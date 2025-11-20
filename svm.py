import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Any
import re

HERO_MAP = {
    # Tanks
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

    # DPS
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

    # Support
    "1020001":  "Mantis",  
    "1023001": "Rocket Raccoon",
    "1028001": "Ultron",
    "1047001": "Jeff The Land Shark",
    "1046001": "Adam Warlock",
    "1050001": "Invisible Woman",
    "1025001": "Cloak & Dagger",
    "1016001": "Loki",
    "1031001": "Luna Snow"
}

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
        #map = match['map']

        team_one_heroes = match['team_one_hero_vector']
        team_one_kills = sum(parse_number(p['kills']) for p in match['team_one'])
        team_one_deaths = sum(parse_number(p['deaths']) for p in match['team_one'])
        team_one_assists = sum(parse_number(p['assists']) for p in match['team_one'])
        team_one_final_hits = sum(parse_number(p['final_hits']) for p in match['team_one'])
        team_one_solo_kills = sum(parse_number(p['solo_kills']) for p in match['team_one'])
        team_one_damage = sum(parse_number(p['damage']) for p in match['team_one'])
        team_one_damage_taken = sum(parse_number(p['damage_taken']) for p in match['team_one'])
        team_one_damage_healed = sum(parse_number(p['damage_healed']) for p in match['team_one'])
        team_one_accuracy = sum(parse_number(p['accuracy']) for p in match['team_one'])

        team_two_heroes = match['team_two_hero_vector']
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
            #'team_one_heroes': team_one_heroes,
            'team_one_kills': team_one_kills,
            'team_one_deaths': team_one_deaths,
            'team_one_assists': team_one_assists,
            'team_one_final_hits': team_one_final_hits,
            'team_one_solo_kills': team_one_solo_kills,
            'team_one_damage': team_one_damage,
            'team_one_damage_taken': team_one_damage_taken,
            'team_one_damage_healed': team_one_damage_healed,
            'team_one_accuracy': team_one_accuracy,
            #'team_two_heroes': team_two_heroes,
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

    df = pd.DataFrame(team_stats)
    #df = df.join(df.pop('team_one_heroes').apply(pd.Series).add_prefix('Team1_Hero_'))
    #df = df.join(df.pop('team_two_heroes').apply(pd.Series).add_prefix('Team2_Hero_'))
    return df

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

def encode_player_heroes(match_data: List[Dict[str, Any]], hero_encoder: Dict[str, int]) -> List[Dict[str, Any]]:
    for m in match_data:
        for team in [m['team_one'], m['team_two']]:
            for player in team:
                hero_vector = [0] * len(hero_encoder)
                total_time = sum(parse_time(hero[1]) for hero in player["heroes_played"])
                for hero in player["heroes_played"]:
                    hero_vector[hero_encoder[hero[0]]] += parse_time(hero[1]) / total_time
                player["hero_vector"] = hero_vector
    return match_data

def encode_team_heroes(match_data: List[Dict[str, Any]], hero_encoder: Dict[str, int]) -> List[Dict[str, Any]]:
    for m in match_data:

        for team in ['team_one', 'team_two']:
            hero_vector = [0] * len(hero_encoder)
            for player in m[team]:
                total_time = sum(parse_time(hero[1]) for hero in player["heroes_played"])
                for hero in player["heroes_played"]:
                    hero_vector[hero_encoder[hero[0]]] += parse_time(hero[1]) / total_time
            m[team+"_hero_vector"] = hero_vector
    return match_data
                

def main():
    match_data = load_match_data('data/match_data_clean.json')
    #hero_encoder = OneHotEncoder(categories=list(HERO_MAP.values()),sparse_output=False, handle_unknown='ignore')
    hero_encoder = {}
    for k in range(len(list(HERO_MAP.items()))):
        hero_encoder[list(HERO_MAP.values())[k]] = k
    encoded_heroes_data = encode_team_heroes(match_data, hero_encoder)
    stats_df = extract_match_stats(match_data)

    print(stats_df.head(1))

    x = stats_df.drop(columns=['is_winner_team_one'])
    y = stats_df['is_winner_team_one'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = SVC(random_state=13)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    main()