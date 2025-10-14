"""Hero statistics visualization module using Sphinx Google Style docstrings."""

import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re


def load_hero_names(filepath: str = 'data/hero_names.json') -> Dict[str, str]:
    """Load hero name mapping from JSON file.

    Args:
        filepath: Path to the hero names JSON file.

    Returns:
        Dictionary mapping hero IDs to hero names.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# Load hero names from external file
HERO_NAMES = load_hero_names()


def get_hero_name(hero_id: str) -> str:
    """Get hero name from hero ID.

    Args:
        hero_id: Hero ID string (e.g., "1018001").

    Returns:
        Hero name or hero ID if not found in mapping.
    """
    return HERO_NAMES.get(hero_id, hero_id)


def get_rank_tier(rank_value: Any) -> str:
    """Convert rank score to tier name.

    Args:
        rank_value: Rank score as string or int.

    Returns:
        Rank tier name (Bronze, Silver, Gold, etc.).
    """
    if rank_value == '-1' or rank_value == -1:
        return 'Unranked'

    try:
        rank = int(str(rank_value).replace(',', ''))
    except:
        return 'Unknown'

    # Rank tier thresholds based on Marvel Rivals ranking system
    if rank < 3300:
        return 'Bronze'
    elif rank < 3600:
        return 'Silver'
    elif rank < 3900:
        return 'Gold'
    elif rank < 4200:
        return 'Platinum'
    elif rank < 4500:
        return 'Diamond'
    elif rank < 4800:
        return 'Grandmaster'
    elif rank < 5000:
        return 'Celestial'
    else:
        return 'Eternity+'


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


def parse_time(time_str: str) -> int:
    """Parse time string to seconds.

    Args:
        time_str: Time string in format like "8m", "42s", "1h30m".

    Returns:
        Total time in seconds.
    """
    total_seconds = 0

    # Extract hours
    hours_match = re.search(r'(\d+)h', time_str)
    if hours_match:
        total_seconds += int(hours_match.group(1)) * 3600

    # Extract minutes
    minutes_match = re.search(r'(\d+)m', time_str)
    if minutes_match:
        total_seconds += int(minutes_match.group(1)) * 60

    # Extract seconds
    seconds_match = re.search(r'(\d+)s', time_str)
    if seconds_match:
        total_seconds += int(seconds_match.group(1))

    return total_seconds


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


def calculate_hero_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Calculate win rate and usage rate for each hero.

    Args:
        match_data: List of match dictionaries from the JSON file.

    Returns:
        DataFrame with columns: hero_id, win_rate, usage_rate, total_games, wins.

    Note:
        Win rate is calculated as (wins / total_games).
        Usage rate is calculated as (total_games / total_matches).
    """
    hero_stats = defaultdict(lambda: {'games': 0, 'wins': 0})
    total_matches = len(match_data)

    for match in match_data:
        winner = match['winner']
        team_one = match['team_one']
        team_two = match['team_two']

        # Process winning team (team_one based on winner=1)
        winning_team = team_one if winner == 1 else team_two
        losing_team = team_two if winner == 1 else team_one

        # Count wins for winning team heroes
        for player in winning_team:
            for hero_id, playtime in player['heroes_played']:
                hero_stats[hero_id]['games'] += 1
                hero_stats[hero_id]['wins'] += 1

        # Count games for losing team heroes
        for player in losing_team:
            for hero_id, playtime in player['heroes_played']:
                hero_stats[hero_id]['games'] += 1

    # Convert to DataFrame
    data = []
    for hero_id, stats in hero_stats.items():
        win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
        usage_rate = stats['games'] / (total_matches * 12) if total_matches > 0 else 0
        data.append({
            'hero_id': hero_id,
            'hero_name': get_hero_name(hero_id),
            'win_rate': win_rate * 100,  # Convert to percentage
            'usage_rate': usage_rate * 100,  # Convert to percentage
            'total_games': stats['games'],
            'wins': stats['wins']
        })

    return pd.DataFrame(data).sort_values('usage_rate', ascending=False)


def extract_team_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract team-level statistics for each match.

    Args:
        match_data: List of match dictionaries from the JSON file.

    Returns:
        DataFrame with columns: match_duration, team_damage, team_kills, team_deaths,
        team_assists, team_final_hits, is_winner.
    """
    team_stats = []

    for match in match_data:
        duration = get_match_duration(match)

        for team_idx, team in enumerate([match['team_one'], match['team_two']], 1):
            team_damage = sum(parse_number(p['damage']) for p in team)
            team_kills = sum(parse_number(p['kills']) for p in team)
            team_deaths = sum(parse_number(p['deaths']) for p in team)
            team_assists = sum(parse_number(p['assists']) for p in team)
            team_final_hits = sum(parse_number(p['final_hits']) for p in team)

            is_winner = (match['winner'] == team_idx)

            team_stats.append({
                'match_duration': duration,
                'team_damage': team_damage,
                'team_kills': team_kills,
                'team_deaths': team_deaths,
                'team_assists': team_assists,
                'team_final_hits': team_final_hits,
                'is_winner': is_winner
            })

    return pd.DataFrame(team_stats)


def extract_player_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract per-player statistics across all matches.

    Args:
        match_data: List of match dictionaries from the JSON file.

    Returns:
        DataFrame with columns: damage, kills, deaths, assists, solo_kills,
        final_hits, damage_taken, damage_healed, rank, rank_tier.
    """
    player_stats = []

    for match in match_data:
        for team in [match['team_one'], match['team_two']]:
            for player in team:
                rank = player['rank']
                player_stats.append({
                    'damage': parse_number(player['damage']),
                    'kills': parse_number(player['kills']),
                    'deaths': parse_number(player['deaths']),
                    'assists': parse_number(player['assists']),
                    'solo_kills': parse_number(player['solo_kills']),
                    'final_hits': parse_number(player['final_hits']),
                    'damage_taken': parse_number(player['damage_taken']),
                    'damage_healed': parse_number(player['damage_healed']),
                    'rank': rank,
                    'rank_tier': get_rank_tier(rank)
                })

    return pd.DataFrame(player_stats)


def create_team_stat_plots(df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create team statistics vs match duration scatterplots.

    Args:
        df: DataFrame containing team statistics.
        output_dir: Directory to save the plot images.

    Note:
        Creates scatterplots showing team stats (damage, kills) vs match duration,
        with winning teams in red and losing teams in blue at 50% opacity.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    stats_to_plot = [
        ('team_damage', 'Total Team Damage'),
        ('team_kills', 'Total Team Kills'),
        ('team_assists', 'Total Team Assists'),
        ('team_final_hits', 'Total Team Final Hits')
    ]

    for stat_col, stat_label in stats_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot losing teams (blue)
        losing_teams = df[~df['is_winner']]
        sns.scatterplot(
            data=losing_teams,
            x='match_duration',
            y=stat_col,
            color='blue',
            alpha=0.5,
            ax=ax,
            label='Losing Teams',
            s=50
        )

        # Plot winning teams (red)
        winning_teams = df[df['is_winner']]
        sns.scatterplot(
            data=winning_teams,
            x='match_duration',
            y=stat_col,
            color='red',
            alpha=0.5,
            ax=ax,
            label='Winning Teams',
            s=50
        )

        ax.set_xlabel('Match Duration (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel(stat_label, fontsize=12, fontweight='bold')
        ax.set_title(
            f'{stat_label} vs Match Duration\n'
            f'Red = Winners, Blue = Losers (Purple = Overlap)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, f'team_{stat_col}_vs_duration.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Created team visualization: {stat_col}')


def create_player_damage_correlations(df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create per-player damage correlation scatterplots.

    Args:
        df: DataFrame containing player statistics.
        output_dir: Directory to save the plot images.

    Note:
        Creates scatterplots showing damage vs kills, solo_kills, and final_hits.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    correlations = [
        ('kills', 'Kills'),
        ('solo_kills', 'Solo Kills'),
        ('final_hits', 'Final Hits')
    ]

    for stat_col, stat_label in correlations:
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.scatterplot(
            data=df,
            x='damage',
            y=stat_col,
            alpha=0.4,
            color='purple',
            ax=ax,
            s=30
        )

        # Calculate correlation coefficient
        corr = df['damage'].corr(df[stat_col])

        ax.set_xlabel('Damage Dealt', fontsize=12, fontweight='bold')
        ax.set_ylabel(stat_label, fontsize=12, fontweight='bold')
        ax.set_title(
            f'Damage vs {stat_label} (Per-Player)\n'
            f'Correlation: {corr:.3f}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, f'player_damage_vs_{stat_col}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Created player correlation: damage vs {stat_col}')


def create_hero_scatterplots(df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create individual scatterplots for each hero showing win rate vs usage rate.

    Args:
        df: DataFrame containing hero statistics.
        output_dir: Directory to save the plot images.

    Note:
        Creates one PNG file per hero in the format: {output_dir}/{hero_id}_winrate_vs_usage.png
        Uses Seaborn styling with a pastel color palette.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")

    for _, row in df.iterrows():
        hero_id = row['hero_id']
        hero_name = row['hero_name']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatterplot showing this hero's position relative to all heroes
        sns.scatterplot(
            data=df,
            x='usage_rate',
            y='win_rate',
            size='total_games',
            sizes=(100, 1000),
            alpha=0.3,
            color='gray',
            ax=ax,
            legend=False
        )

        # Highlight the current hero
        hero_data = df[df['hero_id'] == hero_id]
        sns.scatterplot(
            data=hero_data,
            x='usage_rate',
            y='win_rate',
            size='total_games',
            sizes=(200, 2000),
            color='red',
            ax=ax,
            legend=False,
            edgecolor='darkred',
            linewidth=2
        )

        # Add reference lines
        ax.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='50% Win Rate')
        mean_usage = df['usage_rate'].mean()
        ax.axvline(x=mean_usage, color='green', linestyle='--', alpha=0.5, label='Average Usage')

        # Labels and title
        ax.set_xlabel('Usage Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{hero_name} - Win Rate vs Usage Rate\n'
            f'Win Rate: {hero_data.iloc[0]["win_rate"]:.2f}% | '
            f'Usage Rate: {hero_data.iloc[0]["usage_rate"]:.2f}% | '
            f'Games: {hero_data.iloc[0]["total_games"]}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save figure
        output_path = os.path.join(output_dir, f'{hero_name.replace(" ", "_")}_winrate_vs_usage.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Created visualization for {hero_name}')


def main() -> None:
    """Main execution function.

    Loads match data, calculates hero statistics, and generates visualizations.
    """
    print('Loading match data...')
    match_data = load_match_data('data/match_data.json')
    print(f'Loaded {len(match_data)} matches')

    # Hero statistics
    print('\n=== Calculating hero statistics ===')
    hero_df = calculate_hero_stats(match_data)
    print(f'Analyzed {len(hero_df)} unique heroes')
    hero_df.to_csv('data/hero_statistics.csv', index=False)
    print('Saved statistics to data/hero_statistics.csv')

    # Team statistics
    print('\n=== Extracting team statistics ===')
    team_df = extract_team_stats(match_data)
    print(f'Extracted {len(team_df)} team data points')
    team_df.to_csv('data/team_statistics.csv', index=False)
    print('Saved team statistics to data/team_statistics.csv')

    # Player statistics
    print('\n=== Extracting player statistics ===')
    player_df = extract_player_stats(match_data)
    print(f'Extracted {len(player_df)} player data points')
    player_df.to_csv('data/player_statistics.csv', index=False)
    print('Saved player statistics to data/player_statistics.csv')

    # Create visualizations
    print('\n=== Creating visualizations ===')

    print('\n1. Hero win rate vs usage rate (one plot per hero)...')
    create_hero_scatterplots(hero_df)

    print('\n2. Team statistics vs match duration...')
    create_team_stat_plots(team_df)

    print('\n3. Player damage correlations...')
    create_player_damage_correlations(player_df)

    print('\n=== All visualizations complete! ===')

    # Print summary
    print('\n=== Hero Statistics Summary ===')
    print(hero_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
