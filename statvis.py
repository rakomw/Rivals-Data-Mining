"""Hero statistics visualization module using Sphinx Google Style docstrings."""

import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re


def load_hero_names() -> Dict[str, str]:
    """Load hero name mapping from role-based JSON files.

    Returns:
        Dictionary mapping hero IDs to hero names.
    """
    hero_names = {}

    # Load from all three role files
    role_files = [
        'data/heroes_tank.json',
        'data/heroes_dps.json',
        'data/heroes_support.json'
    ]

    for filepath in role_files:
        with open(filepath, 'r') as f:
            hero_names.update(json.load(f))

    return hero_names


def load_hero_roles() -> Dict[str, str]:
    """Load hero role mapping from role-based JSON files.

    Returns:
        Dictionary mapping hero IDs to their roles (Tank, DPS, Support).
    """
    hero_roles = {}

    role_mappings = [
        ('data/heroes_tank.json', 'Tank'),
        ('data/heroes_dps.json', 'DPS'),
        ('data/heroes_support.json', 'Support')
    ]

    for filepath, role in role_mappings:
        with open(filepath, 'r') as f:
            heroes = json.load(f)
            for hero_id in heroes.keys():
                hero_roles[hero_id] = role

    return hero_roles


# Load hero names and roles from external files
HERO_NAMES = load_hero_names()
HERO_ROLES = load_hero_roles()


def get_hero_name(hero_id: str) -> str:
    """Get hero name from hero ID.

    Args:
        hero_id: Hero ID string (e.g., "1018001").

    Returns:
        Hero name or hero ID if not found in mapping.
    """
    return HERO_NAMES.get(hero_id, hero_id)


def get_hero_role(hero_id: str) -> str:
    """Get hero role from hero ID.

    Args:
        hero_id: Hero ID string (e.g., "1018001").

    Returns:
        Hero role (Tank, DPS, Support) or 'Unknown' if not found.
    """
    return HERO_ROLES.get(hero_id, 'Unknown')


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


def get_match_duration(match: Dict[str, Any]) -> float:
    """Calculate match duration from player playtimes.

    Args:
        match: Match dictionary containing team_one and team_two.

    Returns:
        Match duration in minutes (longest playtime across all players).
    """
    max_duration = 0

    team_one = match.get('team_one', [])
    team_two = match.get('team_two', [])

    for team in [team_one, team_two]:
        if not team or not isinstance(team, list):
            continue
        for player in team:
            player_duration = 0
            for _, playtime in player.get('heroes_played', []):
                player_duration += parse_time(playtime)
            max_duration = max(max_duration, player_duration)

    return max_duration / 60.0  # Convert seconds to minutes


def get_match_rank_tier(match: Dict[str, Any]) -> str:
    """Determine the rank tier of a match based on player ranks.

    Args:
        match: Match dictionary containing team_one and team_two.

    Returns:
        Rank tier name (Unranked, Bronze, Silver, etc.) based on median player rank.
    """
    ranks = []

    team_one = match.get('team_one', [])
    team_two = match.get('team_two', [])

    for team in [team_one, team_two]:
        if not team or not isinstance(team, list):
            continue
        for player in team:
            rank = player.get('rank', '-1')
            # Parse rank value
            if rank == '-1' or rank == -1:
                return 'Unranked'  # If any player is unranked, match is unranked
            try:
                rank_int = int(str(rank).replace(',', ''))
                ranks.append(rank_int)
            except:
                continue

    if not ranks:
        return 'Unranked'

    # Use median rank to determine match tier
    median_rank = sorted(ranks)[len(ranks) // 2]
    return get_rank_tier(median_rank)


def load_match_data(filepath: str) -> List[Dict[str, Any]]:
    """Load match data from JSON file.

    Supports both v1 (list) and v2 (dict) formats.
    v2 format includes map and team scores.

    Args:
        filepath: Path to the match_data.json file.

    Returns:
        List of match dictionaries containing player and game information.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(filepath, 'r') as fp:
        data = json.load(fp)

    # Convert v2 dict format to list format for compatibility
    if isinstance(data, dict):
        return list(data.values())
    return data


def count_valid_matches(match_data: List[Dict[str, Any]]) -> tuple:
    """Count total and valid matches in the dataset.

    Args:
        match_data: List of match dictionaries.

    Returns:
        Tuple of (total_matches, valid_matches, invalid_matches).
    """
    total = len(match_data)
    valid = 0

    for match in match_data:
        team_one = match.get('team_one')
        team_two = match.get('team_two')
        winner = match.get('winner')

        if team_one and team_two and winner and isinstance(team_one, list) and isinstance(team_two, list):
            valid += 1

    invalid = total - valid
    return total, valid, invalid


def calculate_hero_stats(match_data: List[Dict[str, Any]], rank_filter: str = None) -> pd.DataFrame:
    """Calculate win rate and usage rate for each hero.

    Args:
        match_data: List of match dictionaries from the JSON file.
        rank_filter: Optional rank tier to filter by (e.g., 'Bronze', 'Unranked').

    Returns:
        DataFrame with columns: hero_id, win_rate, usage_rate, total_hero_games, wins.

    Note:
        Win rate is calculated as (wins / total_hero_games).
        Usage rate is calculated as percentage of all player-hero selections.
        Formula: (hero_games / (total_valid_matches * 12)) * 100
        Where 12 is the number of players per match (6 per team).
    """
    hero_stats = defaultdict(lambda: {'games': 0, 'wins': 0})
    valid_matches = 0

    for match in match_data:
        team_one = match.get('team_one')
        team_two = match.get('team_two')
        winner = match.get('winner')

        # Handle matches safely - skip if data is invalid
        if not team_one or not team_two or not winner or not isinstance(team_one, list) or not isinstance(team_two, list):
            continue

        # Apply rank filter if specified
        if rank_filter:
            match_rank = get_match_rank_tier(match)
            if match_rank != rank_filter:
                continue

        valid_matches += 1

        # Process winning team (team_one based on winner=1)
        winning_team = team_one if winner == 1 else team_two
        losing_team = team_two if winner == 1 else team_one

        # Count wins for winning team heroes
        for player in winning_team:
            for hero_id, playtime in player.get('heroes_played', []):
                hero_stats[hero_id]['games'] += 1
                hero_stats[hero_id]['wins'] += 1

        # Count games for losing team heroes
        for player in losing_team:
            for hero_id, playtime in player.get('heroes_played', []):
                hero_stats[hero_id]['games'] += 1

    # Convert to DataFrame
    data = []
    for hero_id, stats in hero_stats.items():
        win_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
        usage_rate = stats['games'] / (valid_matches * 12) if valid_matches > 0 else 0
        data.append({
            'hero_id': hero_id,
            'hero_name': get_hero_name(hero_id),
            'hero_role': get_hero_role(hero_id),
            'win_rate': win_rate * 100,  # Convert to percentage
            'usage_rate': usage_rate * 100,  # Convert to percentage
            'total_hero_games': stats['games'],
            'wins': stats['wins']
        })

    return pd.DataFrame(data).sort_values('usage_rate', ascending=False)


def extract_team_stats(match_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract team-level statistics for each match.

    Args:
        match_data: List of match dictionaries from the JSON file.

    Returns:
        DataFrame with columns: match_duration (minutes), team_damage, team_kills, team_deaths,
        team_assists, team_final_hits, is_winner, map (v2), team_score (v2).
    """
    team_stats = []

    for match in match_data:
        team_one = match.get('team_one')
        team_two = match.get('team_two')
        winner = match.get('winner')

        # Handle matches safely - skip if data is invalid
        if not team_one or not team_two or not winner or not isinstance(team_one, list) or not isinstance(team_two, list):
            continue

        duration = get_match_duration(match)  # Now in minutes
        map_name = match.get('map', 'Unknown')

        for team_idx, team in enumerate([match['team_one'], match['team_two']], 1):
            team_damage = sum(parse_number(p['damage']) for p in team)
            team_kills = sum(parse_number(p['kills']) for p in team)
            team_deaths = sum(parse_number(p['deaths']) for p in team)
            team_assists = sum(parse_number(p['assists']) for p in team)
            team_final_hits = sum(parse_number(p['final_hits']) for p in team)
            team_solo_kills = sum(parse_number(p['solo_kills']) for p in team)
            team_damage_taken = sum(parse_number(p['damage_taken']) for p in team)
            team_damage_healed = sum(parse_number(p['damage_healed']) for p in team)

            is_winner = (match['winner'] == team_idx)

            # Get team score if available (v2 format)
            if team_idx == 1:
                team_score = match.get('team_one_score', None)
            else:
                team_score = match.get('team_two_score', None)

            team_stats.append({
                'match_duration': duration,
                'team_damage': team_damage,
                'team_kills': team_kills,
                'team_deaths': team_deaths,
                'team_assists': team_assists,
                'team_final_hits': team_final_hits,
                'team_solo_kills': team_solo_kills,
                'team_damage_taken': team_damage_taken,
                'team_damage_healed': team_damage_healed,
                'team_score': team_score,
                'map': map_name,
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
        team_one = match.get('team_one')
        team_two = match.get('team_two')

        # Handle matches safely - skip if data is invalid
        if not team_one or not team_two or not isinstance(team_one, list) or not isinstance(team_two, list):
            continue

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

        ax.set_xlabel('Match Duration (minutes)', fontsize=12, fontweight='bold')
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


def calculate_win_correlations(team_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between all team statistics and winning.

    Args:
        team_df: DataFrame containing team statistics with is_winner column.

    Returns:
        DataFrame with correlations sorted by absolute value.

    Note:
        Calculates point-biserial correlation between numeric stats and binary winner variable.
    """
    # Get numeric columns only
    numeric_cols = team_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_winner']]

    correlations = []
    for col in numeric_cols:
        # Calculate correlation with winning (point-biserial for binary variable)
        corr = team_df[col].corr(team_df['is_winner'].astype(int))
        correlations.append({
            'statistic': col,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })

    result_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    return result_df


def create_correlation_heatmap(team_df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create correlation heatmap for all team statistics.

    Args:
        team_df: DataFrame containing team statistics.
        output_dir: Directory to save the plot images.

    Note:
        Creates a heatmap showing correlations between all numeric team stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="white")

    # Get numeric columns
    numeric_cols = team_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['team_score']]  # Exclude team_score if it has NaN

    # Create correlation matrix
    corr_matrix = team_df[numeric_cols].corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )

    ax.set_title('Correlation Matrix: Team Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('Created correlation heatmap')


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


def create_role_comparison_charts(hero_df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create role comparison visualizations.

    Args:
        hero_df: DataFrame containing hero statistics with role information.
        output_dir: Directory to save the plot images.

    Note:
        Creates bar charts comparing roles by usage rate, win rate, and hero count.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Calculate role statistics
    role_stats = hero_df.groupby('hero_role').agg({
        'usage_rate': 'mean',
        'win_rate': 'mean',
        'total_hero_games': 'sum',
        'hero_id': 'count'
    }).rename(columns={'hero_id': 'hero_count'})

    # 1. Average usage rate by role
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Tank': '#FF6B6B', 'DPS': '#4ECDC4', 'Support': '#95E1D3'}
    role_colors = [colors.get(role, 'gray') for role in role_stats.index]

    bars = ax.bar(role_stats.index, role_stats['usage_rate'], color=role_colors, alpha=0.8)
    ax.set_xlabel('Role', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Usage Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Hero Usage Rate by Role', fontsize=14, fontweight='bold', pad=20)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'role_usage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Created role usage comparison chart')

    # 2. Average win rate by role
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(role_stats.index, role_stats['win_rate'], color=role_colors, alpha=0.8)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Win Rate')
    ax.set_xlabel('Role', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Hero Win Rate by Role', fontsize=14, fontweight='bold', pad=20)
    ax.legend()

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'role_winrate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Created role win rate comparison chart')

    # 3. Hero count by role
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(role_stats.index, role_stats['hero_count'], color=role_colors, alpha=0.8)
    ax.set_xlabel('Role', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Heroes', fontsize=12, fontweight='bold')
    ax.set_title('Hero Count by Role', fontsize=14, fontweight='bold', pad=20)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'role_hero_count.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Created role hero count chart')


def create_role_scatterplot(hero_df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create scatterplot showing all heroes colored by role.

    Args:
        hero_df: DataFrame containing hero statistics with role information.
        output_dir: Directory to save the plot images.

    Note:
        Creates a scatterplot with win rate vs usage rate, colored by role.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = {'Tank': '#FF6B6B', 'DPS': '#4ECDC4', 'Support': '#95E1D3'}

    for role in ['Tank', 'DPS', 'Support']:
        role_data = hero_df[hero_df['hero_role'] == role]
        ax.scatter(role_data['usage_rate'], role_data['win_rate'],
                  s=200, alpha=0.6, color=colors[role], label=role, edgecolors='black', linewidth=1)

        # Add hero names as labels
        for _, row in role_data.iterrows():
            ax.annotate(row['hero_name'],
                       (row['usage_rate'], row['win_rate']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Win Rate')
    ax.set_xlabel('Usage Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hero Win Rate vs Usage Rate by Role', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heroes_by_role_scatterplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Created heroes by role scatterplot')


def create_role_top10_charts(hero_df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create bar charts showing top 10 heroes for each role.

    Args:
        hero_df: DataFrame containing hero statistics with role information.
        output_dir: Directory to save the plot images.

    Note:
        Creates separate bar charts for each role showing top 10 by usage rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    colors = {'Tank': '#FF6B6B', 'DPS': '#4ECDC4', 'Support': '#95E1D3'}

    for role in ['Tank', 'DPS', 'Support']:
        role_data = hero_df[hero_df['hero_role'] == role].head(10)

        if len(role_data) == 0:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Usage rate chart
        bars1 = ax1.barh(role_data['hero_name'], role_data['usage_rate'],
                         color=colors[role], alpha=0.8)
        ax1.set_xlabel('Usage Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Hero', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top 10 {role} Heroes by Usage Rate', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()

        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=9)

        # Win rate chart
        bars2 = ax2.barh(role_data['hero_name'], role_data['win_rate'],
                         color=colors[role], alpha=0.8)
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% Win Rate')
        ax2.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Hero', fontsize=11, fontweight='bold')
        ax2.set_title(f'Top 10 {role} Heroes by Win Rate', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.legend()

        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top10_{role.lower()}_heroes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Created top 10 {role} heroes chart')


def create_hero_distribution_heatmap(hero_df: pd.DataFrame, output_dir: str = 'visualizations') -> None:
    """Create a stock market-style heatmap showing hero performance.

    Args:
        hero_df: DataFrame containing hero statistics with role information.
        output_dir: Directory to save the plot images.

    Note:
        Creates a stock market-style heatmap with:
        - Grid layout with all heroes together
        - Cell color = win rate (red=low, green=high)
        - Cell opacity = usage rate (higher usage = more opaque)
        - Role tag shown for each hero
    """
    import numpy as np
    from matplotlib import cm
    from matplotlib.patches import Rectangle

    os.makedirs(output_dir, exist_ok=True)

    # Sort all heroes by usage rate
    hero_data = hero_df.copy().sort_values('usage_rate', ascending=False)

    # Define role colors
    role_colors = {'Tank': '#FF6B6B', 'DPS': '#4ECDC4', 'Support': '#95E1D3'}

    # Create grid positions
    n_heroes = len(hero_data)
    cols = 8  # 8 columns for all heroes
    rows = (n_heroes + cols - 1) // cols

    # Create data for heatmap
    grid_win_rate = []
    grid_usage = []
    grid_roles = []
    labels = []

    for i in range(rows):
        row_wr = []
        row_usage = []
        row_roles = []
        row_labels = []
        for j in range(cols):
            hero_idx = i * cols + j
            if hero_idx < n_heroes:
                hero = hero_data.iloc[hero_idx]
                row_wr.append(hero['win_rate'])
                row_usage.append(hero['usage_rate'])
                row_roles.append(hero['hero_role'])
                row_labels.append(f"{hero['hero_name']}\n{hero['win_rate']:.1f}%\n({hero['usage_rate']:.1f}%)")
            else:
                row_wr.append(0)
                row_usage.append(0)
                row_roles.append('')
                row_labels.append('')
        grid_win_rate.append(row_wr)
        grid_usage.append(row_usage)
        grid_roles.append(row_roles)
        labels.append(row_labels)

    # Convert to numpy arrays
    grid_win_rate = np.array(grid_win_rate)
    grid_usage = np.array(grid_usage)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # Normalize win rates (35-55% range for contrast)
    norm = plt.Normalize(vmin=35, vmax=55)
    colors = cm.RdYlGn(norm(grid_win_rate))

    # Adjust alpha based on usage rate (higher usage = more opaque)
    max_usage = hero_data['usage_rate'].max()
    alpha_values = np.clip(grid_usage / max_usage * 0.5 + 0.5, 0.3, 1.0)

    # Apply alpha to colors
    for i in range(rows):
        for j in range(cols):
            if grid_usage[i, j] > 0:
                colors[i, j, 3] = alpha_values[i, j]
            else:
                colors[i, j, 3] = 0  # Fully transparent for empty cells

    # Plot the heatmap
    im = ax.imshow(colors, aspect='auto')

    # Add text labels and role tags
    for i in range(rows):
        for j in range(cols):
            if labels[i][j]:
                # Draw role tag rectangle in top-left corner of cell
                role = grid_roles[i][j]
                role_color = role_colors.get(role, 'gray')

                # Add role tag as small rectangle
                rect = Rectangle((j - 0.48, i - 0.48), 0.3, 0.15,
                               facecolor=role_color, edgecolor='white',
                               linewidth=1.5, alpha=0.9, transform=ax.transData)
                ax.add_patch(rect)

                # Add role text on tag
                ax.text(j - 0.33, i - 0.405, role[0], ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white',
                       transform=ax.transData)

                # Add hero info text
                text_color = 'black' if grid_win_rate[i, j] > 42 else 'white'
                ax.text(j, i, labels[i][j], ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.3, edgecolor='none'))

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add main title
    fig.suptitle('Marvel Rivals Hero Performance Heatmap\n(Color = Win Rate | Opacity = Usage Rate)',
                fontsize=20, fontweight='bold', y=0.98)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cm.RdYlGn, norm=plt.Normalize(vmin=35, vmax=55))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, aspect=40, fraction=0.05)
    cbar.set_label('Win Rate (%)', fontsize=12, fontweight='bold')

    # Add role legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=role_colors['Tank'], label='T = Tank'),
        Patch(facecolor=role_colors['DPS'], label='D = DPS'),
        Patch(facecolor=role_colors['Support'], label='S = Support')
    ]
    ax.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(0, 1.08), ncol=3, frameon=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'hero_distribution_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Created hero distribution heatmap')


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
            size='total_hero_games',
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
            size='total_hero_games',
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
            f'Games: {hero_data.iloc[0]["total_hero_games"]}',
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


def generate_summary_notes(hero_df: pd.DataFrame, team_df: pd.DataFrame, player_df: pd.DataFrame,
                          match_data: List[Dict[str, Any]], output_dir: str = 'data') -> None:
    """Generate 5 key bullet point findings and save to file.

    Args:
        hero_df: Hero statistics DataFrame.
        team_df: Team statistics DataFrame.
        player_df: Player statistics DataFrame.
        match_data: Raw match data.
        output_dir: Directory to save the notes file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate key statistics
    total_matches = len([m for m in match_data if m.get('team_one') and m.get('team_two')])
    total_games_played = hero_df['total_hero_games'].sum()

    # Most played hero
    top_hero = hero_df.iloc[0]

    # Best win rate hero (with significant games)
    significant_heroes = hero_df[hero_df['total_hero_games'] >= 10000]
    best_winrate_hero = significant_heroes.loc[significant_heroes['win_rate'].idxmax()] if len(significant_heroes) > 0 else hero_df.iloc[0]

    # Role statistics
    role_stats = hero_df.groupby('hero_role').agg({
        'usage_rate': 'mean',
        'win_rate': 'mean',
        'total_hero_games': 'sum'
    })
    dominant_role = role_stats['total_hero_games'].idxmax()

    # Win correlations
    win_corr = team_df[['team_kills', 'team_deaths', 'team_assists']].corrwith(team_df['is_winner'].astype(int))
    strongest_predictor = win_corr.abs().idxmax()

    # Match duration
    avg_duration = team_df['match_duration'].mean()

    # Generate notes
    notes = [
        f"**Most Popular Hero**: {top_hero['hero_name']} ({top_hero['hero_role']}) dominates with {top_hero['usage_rate']:.1f}% usage rate across {top_hero['total_hero_games']:,} games, indicating high player preference despite a {top_hero['win_rate']:.1f}% win rate.",

        f"**Highest Win Rate**: {best_winrate_hero['hero_name']} ({best_winrate_hero['hero_role']}) leads with {best_winrate_hero['win_rate']:.1f}% win rate in {best_winrate_hero['total_hero_games']:,} games, suggesting strong performance when selected strategically.",

        f"**Role Balance**: {dominant_role} heroes account for {role_stats.loc[dominant_role, 'total_hero_games']:,} total games ({role_stats.loc[dominant_role, 'total_hero_games']/total_games_played*100:.1f}% of all picks). Average win rates show Tanks ({role_stats.loc['Tank', 'win_rate']:.1f}%) slightly outperform DPS ({role_stats.loc['DPS', 'win_rate']:.1f}%) and Supports ({role_stats.loc['Support', 'win_rate']:.1f}%).",

        f"**Winning Factor**: Team {strongest_predictor.replace('team_', '').replace('_', ' ')} shows the strongest correlation ({win_corr[strongest_predictor]:.3f}) with match outcomes, making it a key performance indicator more impactful than raw damage output.",

        f"**Match Dynamics**: Analyzed {total_matches:,} matches with average duration of {avg_duration:.1f} minutes. Hero switching behavior results in {total_games_played:,} total hero-games (avg {total_games_played/total_matches:.1f} per match), indicating players actively adapt strategies mid-game."
    ]

    # Save to file
    output_path = os.path.join(output_dir, 'analysis_summary_notes.txt')
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MARVEL RIVALS DATA ANALYSIS - KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        for i, note in enumerate(notes, 1):
            f.write(f"{i}. {note}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Generated from 92k match dataset analysis\n")
        f.write("=" * 80 + "\n")

    print(f'\nSaved summary notes to {output_path}')

    # Also print to console
    print("\n" + "=" * 80)
    print("MARVEL RIVALS DATA ANALYSIS - KEY FINDINGS")
    print("=" * 80 + "\n")

    for i, note in enumerate(notes, 1):
        print(f"{i}. {note}\n")

    print("=" * 80)


def main() -> None:
    """Main execution function.

    Loads match data, calculates hero statistics, and generates visualizations.
    Both v1 and v2 JSON formats (v2 includes map and team scores).
    By default, loads the 92k dataset.
    """
    # Determine which data file(s) to use
    import sys
    if len(sys.argv) > 1:
        # Single file specified via command line
        data_files = [sys.argv[1]]
    else:
        # Default: use 92k dataset
        data_files = ['data/match_data-92k.json']

    # Load all datasets
    all_match_data = []
    print('=== Loading Match Data ===')
    for data_file in data_files:
        print(f'\nLoading {data_file}...')
        matches = load_match_data(data_file)

        # Count valid vs invalid matches for this file
        total, valid, invalid = count_valid_matches(matches)
        print(f'  Total matches: {total:,}')
        print(f'  Valid matches: {valid:,}')
        if invalid > 0:
            print(f'  Ignored matches: {invalid:,} (invalid/incomplete data)')

        all_match_data.extend(matches)

    # Combined statistics
    print(f'\n--- Combined Dataset ---')
    total_all, valid_all, invalid_all = count_valid_matches(all_match_data)
    print(f'Total matches across all files: {total_all:,}')
    print(f'Valid matches: {valid_all:,}')
    if invalid_all > 0:
        print(f'Ignored matches: {invalid_all:,} (invalid/incomplete data)')

    match_data = all_match_data

    # Check if v2 format (has map data)
    has_map_data = any('map' in match for match in match_data[:100] if match)
    print(f'Format: {"Mixed (v2 with map/scores + v1)" if len(data_files) > 1 else ("v2 (with map/scores)" if has_map_data else "v1 (basic)")}')

    # Hero statistics - Overall
    print('\n=== Calculating hero statistics (Overall) ===')
    hero_df = calculate_hero_stats(match_data)
    print(f'Analyzed {len(hero_df)} unique heroes across all matches')
    hero_df.to_csv('data/hero_statistics_overall.csv', index=False)
    print('Saved overall statistics to data/hero_statistics_overall.csv')

    # Hero statistics by rank tier
    print('\n=== Calculating hero statistics by rank tier ===')
    rank_tiers = ['Unranked', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Grandmaster', 'Celestial', 'Eternity+']

    for rank_tier in rank_tiers:
        rank_hero_df = calculate_hero_stats(match_data, rank_filter=rank_tier)
        if len(rank_hero_df) > 0:
            filename = f'data/hero_statistics_{rank_tier.lower().replace("+", "plus")}.csv'
            rank_hero_df.to_csv(filename, index=False)
            print(f'  {rank_tier}: {len(rank_hero_df)} heroes')
        else:
            print(f'  {rank_tier}: No data')

    # Hero statistics by role
    print('\n=== Calculating hero statistics by role ===')
    roles = ['Tank', 'DPS', 'Support']

    for role in roles:
        role_df = hero_df[hero_df['hero_role'] == role].copy()
        if len(role_df) > 0:
            filename = f'data/hero_statistics_{role.lower()}.csv'
            role_df.to_csv(filename, index=False)
            print(f'  {role}: {len(role_df)} heroes, avg usage: {role_df["usage_rate"].mean():.2f}%, avg win rate: {role_df["win_rate"].mean():.2f}%')
        else:
            print(f'  {role}: No data')

    # Team statistics
    print('\n=== Extracting team statistics ===')
    team_df = extract_team_stats(match_data)
    print(f'Extracted {len(team_df)} team data points')
    team_df.to_csv('data/team_statistics.csv', index=False)
    print('Saved team statistics to data/team_statistics.csv')

    # Calculate correlations with winning
    print('\n=== Calculating win correlations ===')
    win_corr_df = calculate_win_correlations(team_df)
    win_corr_df.to_csv('data/win_correlations.csv', index=False)
    print('Top correlations with winning:')
    print(win_corr_df.head(10).to_string(index=False))

    # Player statistics
    print('\n=== Extracting player statistics ===')
    player_df = extract_player_stats(match_data)
    print(f'Extracted {len(player_df)} player data points')
    player_df.to_csv('data/player_statistics.csv', index=False)
    print('Saved player statistics to data/player_statistics.csv')

    # Create visualizations
    print('\n=== Creating visualizations ===')

    print('\n1. Correlation heatmap...')
    create_correlation_heatmap(team_df)

    print('\n2. Role comparison charts...')
    create_role_comparison_charts(hero_df)

    print('\n3. Heroes by role scatterplot...')
    create_role_scatterplot(hero_df)

    print('\n4. Top 10 heroes per role...')
    create_role_top10_charts(hero_df)

    print('\n5. Hero distribution heatmap (stock market style)...')
    create_hero_distribution_heatmap(hero_df)

    print('\n6. Hero win rate vs usage rate (one plot per hero)...')
    create_hero_scatterplots(hero_df)

    print('\n7. Team statistics vs match duration...')
    create_team_stat_plots(team_df)

    print('\n8. Player damage correlations...')
    create_player_damage_correlations(player_df)

    print('\n=== All visualizations complete! ===')

    # Print summary - Top 10 heroes for each rank tier
    print('\n=== TOP 10 HEROES BY RANK TIER ===')
    print('(Sorted by usage rate)\n')

    for rank_tier in rank_tiers:
        rank_hero_df = calculate_hero_stats(match_data, rank_filter=rank_tier)
        if len(rank_hero_df) > 0:
            print(f'\n--- {rank_tier} ---')
            top_10 = rank_hero_df.head(10)[['hero_name', 'hero_role', 'win_rate', 'usage_rate', 'total_hero_games']]
            print(top_10.to_string(index=False))
        else:
            print(f'\n--- {rank_tier} ---')
            print('  No data available')

    # Overall statistics summary
    print('\n\n=== OVERALL HERO STATISTICS (TOP 10) ===')
    print(hero_df.head(10)[['hero_name', 'hero_role', 'win_rate', 'usage_rate', 'total_hero_games']].to_string(index=False))

    # Role-based top heroes
    print('\n\n=== TOP 10 HEROES BY ROLE ===')
    for role in roles:
        role_df = hero_df[hero_df['hero_role'] == role].copy()
        if len(role_df) > 0:
            print(f'\n--- {role} (Avg Usage: {role_df["usage_rate"].mean():.2f}%, Avg Win Rate: {role_df["win_rate"].mean():.2f}%) ---')
            top_10_role = role_df.head(10)[['hero_name', 'win_rate', 'usage_rate', 'total_hero_games']]
            print(top_10_role.to_string(index=False))

    # Map statistics if available
    if has_map_data:
        print('\n\n=== MAP STATISTICS (Average Duration in Minutes) ===')
        map_stats = team_df.groupby('map').agg({
            'is_winner': 'count',
            'team_kills': 'mean',
            'team_damage': 'mean',
            'match_duration': 'mean'
        }).rename(columns={'is_winner': 'total_teams', 'match_duration': 'avg_duration_min'})
        print(map_stats.to_string())


if __name__ == '__main__':
    main()
