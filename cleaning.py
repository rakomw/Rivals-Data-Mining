import json
from copy import deepcopy
"""
By Ken Douglas

Classifying the Imported Json data, Find the format and how to recapture the data for cleaning into a .csv



|| List of hero Idenitifying Codes by image number
----Tanks
Emma Frost = 1053001
Thor = 1039001
Doctor Strange = 1018001
Captain America = 1022001
Angela = 1056001
Hulk = 1011001
Peni Parker = 1042001
Venom = 1035001
Groot = 1027001
The Thing = 1051001
Magneto = 1037001

----DPS
Magik = 1029001
Storm = 1015001
Mister Fantastic = 1040001
Iron Fist = 1052001
Black Panther = 1026001
Hela = 1024001
Iron Man = 1034001
Star Lord = 1043001
Namor = 1045001
Scarlet Witch = 1038001
Human Torch = 1017001
Spider Man = 1036001
Moon Knight = 1030001
Winter Soldier = 1041001
Psylocke = 1048001
Phoenix = 1054001
Blade = 1044001
Hawkeye = 1021001
Wolverine = 1049001
The Punisher = 1014001
Squirrel Girl = 1032001
Black Widow = 1033001
Daredevil = 1055001

----Support
Mantis = 102001
Rocket Raccoon = 1023001
Ultron = 1028001
Jeff The Land Shark = 1047001
Adam Warlock = 1046001
Invisible Woman = 1050001
Cloak & Dagger = 1025001
Loki = 1016001
Luna Snow = 1031001


|| Rank Elo Conversion
Eternity/One Above All: 5000+
Celestial 1: 5000
Celestial 2: 4900
Celestial 3: 4800
Grandmaster 1: 4700
Grandmaster 2: 4600
Grandmaster 3: 4500
Diamond 1: 4400
Diamond 2: 4300
Diamond 3: 4200
Platinum 1: 4100
Platinum 2: 4000
Platinum 3: 3900
Gold 1: 3800
Gold 2: 3700
Gold 3: 3600
Silver 1: 3500
Silver 2: 3400
Silver 3: 3300
Bronze 1: 3200
Bronze 2: 3100
Bronze 3: 3000



"""

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

RANK_MAP = {
    "-1": "Unranked",
    "3000": "Bronze 3",
    "3100": "Bronze 2",
    "3200": "Bronze 1",
    "3300": "Silver 3",
    "3400": "Silver 2",
    "3500": "Silver 1",
    "3600": "Gold 3",
    "3700": "Gold 2",
    "3800": "Gold 1",
    "3900": "Platinum 3",
    "4000": "Platinum 2",
    "4100": "Platinum 1",
    "4200": "Diamond 3",
    "4300": "Diamond 2",
    "4400": "Diamond 1",
    "4500": "Grandmaster 3",
    "4600": "Grandmaster 2",
    "4700": "Grandmaster 1",
    "4800": "Celestial 3",
    "4900": "Celestial 2",
    "5000": "Celestial 1",
    "5100": "Eternity/One Above All"
}

HEROES_COUNTED = 1


def _minutes_from_time_str(time_str: str) -> float:
    """Convert a time token like '14m' or '42s' into minutes (float, 2 decimals)."""
    if not isinstance(time_str, str):
        return 0.0
    t = time_str.strip().lower()
    if t.endswith("m"):
        try:
            return float(t[:-1])
        except ValueError:
            return 0.0
    if t.endswith("s"):
        try:
            sec = float(t[:-1])
        except ValueError:
            return 0.0
        return round(sec / 60.0, 2)
    # Unknown format -> 0
    return 0.0


# Helper to clean rank values to tier label
def clean_rank_value(rank_value) -> str:
    """Map a raw numeric rank (e.g. '4,851') to a tier label using RANK_MAP.
    Floors to nearest 100. <3000 -> 3000. >=5000 -> 5100 (Eternity/One Above All).
    Returns the label string.
    """
    if rank_value is None:
        return "Unknown"
    try:
        n = int(str(rank_value).replace(',', '').strip())
    except ValueError:
        return f"Unknown_{rank_value}"

    if n == -1:
        return "Unranked"

    if n >= 5100:
        key = "5100"
    else:
        # floor to nearest hundred
        floored = n - (n % 100)
        if floored < 3000:
            floored = 3000
        key = str(floored)

    return RANK_MAP.get(key, f"Unknown_{n}")


def clean_heroes_played(heroes_played, hero_map=HERO_MAP, choose_top=HEROES_COUNTED):
    """Return a new list where hero IDs are replaced with names and times unified to minutes.

    Input example: [["1053001", "8m"], ["1039001", "42s"]]
    Output example: [["Emma Frost", "8.0m"], ["Thor", "0.7m"]]
    """
    cleaned = []
    if not heroes_played:
        return cleaned
    for pair in heroes_played:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        hero_id, time_str = pair
        hero_name = hero_map.get(str(hero_id), f"Unknown_{hero_id}")
        minutes = _minutes_from_time_str(time_str)
        cleaned.append([hero_name, f"{minutes:.2f}m"])  # keep 2 decimals for consistency
    if choose_top > 0:
        cleaned.sort(key=lambda pair: _minutes_from_time_str(pair[1]), reverse=True)
        if choose_top > len(cleaned):
            choose_top = len(cleaned)
        cleaned = cleaned[0:choose_top]
    return cleaned



def clean_player(player: dict) -> dict:
    """Return a new player dict with cleaned `heroes_played`. Adjust string numbers to int"""
    if not isinstance(player, dict):
        return player
    p = deepcopy(player)
    p["heroes_played"] = clean_heroes_played(player.get("heroes_played", []))
    raw_rank = player.get("rank")
    p["rank"] = clean_rank_value(raw_rank)

    p["damage"] = str(player.get("damage", "0")).replace(',', '')
    p["damage_taken"] = str(player.get("damage_taken", "0")).replace(',', '')
    p["damage_healed"] = str(player.get("damage_healed", "0")).replace(',', '')
    return p


# --- Team filtering helpers ---

def _player_total_minutes(player: dict) -> float:
    """Sum total minutes from the already-cleaned heroes_played list (e.g., "8.00m")."""
    total = 0.0
    for pair in player.get("heroes_played", []) or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        _, minutes_str = pair
        if isinstance(minutes_str, str) and minutes_str.endswith("m"):
            try:
                total += float(minutes_str[:-1])
            except ValueError:
                pass
    return total


def _filter_and_trim_team(team_players: list, max_players: int = 6) -> list:
    """Remove players with 0 total minutes; if > max_players remain, keep top by minutes.
    Returns a NEW list without mutating input.
    """
    if not isinstance(team_players, list):
        return []
    # Filter out zero-minute players
    filtered = [p for p in team_players if _player_total_minutes(p) > 0.0]
    # If more than max_players, keep the top N by total minutes
    if len(filtered) > max_players:
        filtered = sorted(filtered, key=_player_total_minutes, reverse=True)[:max_players]
    return filtered


import random

def clean_match(match: dict) -> dict:
    """Return a new match dict with cleaned teams.
    - Cleans players (hero IDs + time normalization + rank label)
    - Removes players with 0 minutes total
    - If a team has >6 players with minutes, keeps the top 6 by minutes
    - Randomly flips teams and winner (50%) for balanced training
    """
    if not isinstance(match, dict):
        return None
    elif match.get("team_one") is None or match.get("team_two") is None:
        return None
    elif match.get("team_one") == [] or match.get("team_two") == []:
        return None

    m = deepcopy(match)
    m["team_one"] = [clean_player(p) for p in match.get("team_one", [])]
    m["team_two"] = [clean_player(p) for p in match.get("team_two", [])]

    # Apply filtering: drop 0-minute players; cap at 6 by total minutes
    m["team_one"] = _filter_and_trim_team(m["team_one"], max_players=6)
    m["team_two"] = _filter_and_trim_team(m["team_two"], max_players=6)

    # Randomly flip match teams for balanced data
    winner = match.get("winner")
    if isinstance(winner, int) and winner in (1, 2):
        if random.choice([True, False]):  # 50% chance to flip teams
            # Swap teams
            temp = m["team_one"]
            m["team_one"] = m["team_two"]
            m["team_two"] = temp

            # Flip winner (1 ↔ 2)
            m["winner"] = 2 if winner == 1 else 1
        else:
            # Keep winner as is
            m["winner"] = winner
    return m


def clean_all_matches(matches):
    """Return a new list of matches with hero IDs replaced and times normalized to minutes."""
    if not isinstance(matches, list):
        return matches
    cleaned = [clean_match(mtch) for mtch in matches]
    return [m for m in cleaned if m is not None]

def main():
    # Load the raw JSON list of matches
    with open("data/match_data-v2.json", "r") as f:
        matches = json.load(f)

    matches = list(matches.values())
    print(len(matches))
    # Show a small BEFORE sample (first player's heroes in first match)
    try:
        sample_before = matches[0]["team_one"][0]["heroes_played"]
    except Exception as e:
        print("Could not read sample BEFORE heroes_played:", e)
        sample_before = None

    # Clean all matches
    cleaned = clean_all_matches(matches)
    print(len(cleaned))

    # Show a small AFTER sample to verify transformation
    try:
        sample_after = cleaned[0]["team_one"][0]["heroes_played"]
    except Exception as e:
        print("Could not read sample AFTER heroes_played:", e)
        sample_after = None

    print("BEFORE:", sample_before)
    print("AFTER:", sample_after)

    # Write cleaned results to a new file for inspection
    out_path = "data/match_data_one_hero.json"
    with open(out_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"Wrote cleaned JSON to {out_path}")

if __name__ == "__main__":
    main()
