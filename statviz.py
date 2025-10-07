"""Hero statistics visualization module using Sphinx Google Style docstrings."""

import json
from collections import defaultdict
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


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
            'win_rate': win_rate * 100,  # Convert to percentage
            'usage_rate': usage_rate * 100,  # Convert to percentage
            'total_games': stats['games'],
            'wins': stats['wins']
        })

    return pd.DataFrame(data).sort_values('usage_rate', ascending=False)


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
            f'{hero_id} - Win Rate vs Usage Rate\n'
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
        output_path = os.path.join(output_dir, f'{hero_id}_winrate_vs_usage.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Created visualization for {hero_id}')


def main() -> None:
    """Main execution function.

    Loads match data, calculates hero statistics, and generates visualizations.
    """
    print('Loading match data...')
    match_data = load_match_data('data/match_data.json')
    print(f'Loaded {len(match_data)} matches')

    print('Calculating hero statistics...')
    hero_df = calculate_hero_stats(match_data)
    print(f'Analyzed {len(hero_df)} unique heroes')

    # Save statistics to CSV
    hero_df.to_csv('data/hero_statistics.csv', index=False)
    print('Saved statistics to data/hero_statistics.csv')

    print('Creating visualizations...')
    create_hero_scatterplots(hero_df)
    print('Visualizations complete!')

    # Print summary
    print('\n=== Hero Statistics Summary ===')
    print(hero_df.to_string(index=False))


if __name__ == '__main__':
    main()
