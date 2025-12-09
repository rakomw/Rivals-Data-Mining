"""
Minimal RAG pipeline for Marvel Rivals match data.
Uses Ollama for embeddings/LLM and ChromaDB for vector storage.
"""
import csv
import ollama
import chromadb

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.1"
CSV_PATH = "data/stats_and_heroes.csv"
TOP_K = 100

# Hero names in order (matches Team1_Hero_0 to Team1_Hero_42)
HERO_NAMES = [
    "Emma Frost", "Thor", "Doctor Strange", "Captain America", "Angela",
    "Hulk", "Peni Parker", "Venom", "Groot", "The Thing", "Magneto",
    "Magik", "Storm", "Mister Fantastic", "Iron Fist", "Black Panther",
    "Hela", "Iron Man", "Star-Lord", "Namor", "Scarlet Witch", "Human Torch",
    "Spider-Man", "Moon Knight", "Winter Soldier", "Psylocke", "Phoenix",
    "Blade", "Hawkeye", "Wolverine", "The Punisher", "Squirrel Girl",
    "Black Widow", "Daredevil", "Mantis", "Rocket Raccoon", "Ultron",
    "Jeff The Land Shark", "Adam Warlock", "Invisible Woman", "Cloak & Dagger",
    "Loki", "Luna Snow"
]


def get_heroes_from_row(row: dict, team: str) -> list[str]:
    """Extract hero names with >0.1 play rate for a team."""
    heroes = []
    for i, name in enumerate(HERO_NAMES):
        key = f"{team}_Hero_{i}"
        if key in row and float(row[key]) > 0.1:
            heroes.append(name)
    return heroes


def load_documents(csv_path: str, limit: int = 5000) -> tuple[list[str], list[str]]:
    """Load CSV rows as readable text documents with meta statistics.

    Returns:
        tuple: (all_docs, meta_docs) where meta_docs are the aggregate statistics
    """
    docs = []
    stats = {
        "total": 0, "t1_wins": 0, "t2_wins": 0,
        "total_duration": 0, "total_damage": 0, "total_kills": 0,
        "total_deaths": 0, "total_assists": 0, "total_healed": 0,
        "hero_wins": {name: 0 for name in HERO_NAMES},
        "hero_picks": {name: 0 for name in HERO_NAMES},
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break

            # Track aggregate stats
            stats["total"] += 1
            stats["total_duration"] += float(row["match_duration"])
            stats["total_damage"] += float(row["team_one_damage"]) + float(row["team_two_damage"])
            stats["total_kills"] += float(row["team_one_kills"]) + float(row["team_two_kills"])
            stats["total_deaths"] += float(row["team_one_deaths"]) + float(row["team_two_deaths"])
            stats["total_assists"] += float(row["team_one_assists"]) + float(row["team_two_assists"])
            stats["total_healed"] += float(row["team_one_damage_healed"]) + float(row["team_two_damage_healed"])

            t1_wins = row["is_winner_team_one"] == "True"
            if t1_wins:
                stats["t1_wins"] += 1
            else:
                stats["t2_wins"] += 1

            # Track hero stats
            t1_heroes = get_heroes_from_row(row, "Team1")
            t2_heroes = get_heroes_from_row(row, "Team2")
            for hero in t1_heroes:
                stats["hero_picks"][hero] += 1
                if t1_wins:
                    stats["hero_wins"][hero] += 1
            for hero in t2_heroes:
                stats["hero_picks"][hero] += 1
                if not t1_wins:
                    stats["hero_wins"][hero] += 1

            # Build readable document
            winner = "Team 1" if t1_wins else "Team 2"
            doc = f"""Match {i+1}: {winner} wins.
Duration: {int(float(row['match_duration']))} seconds.
Team 1 heroes: {', '.join(t1_heroes) or 'Unknown'}. Kills: {row['team_one_kills']}, Deaths: {row['team_one_deaths']}, Damage: {row['team_one_damage']}.
Team 2 heroes: {', '.join(t2_heroes) or 'Unknown'}. Kills: {row['team_two_kills']}, Deaths: {row['team_two_deaths']}, Damage: {row['team_two_damage']}."""
            docs.append(doc)

    # Compute averages
    n = stats["total"]
    avg_duration = stats["total_duration"] / n if n > 0 else 0
    avg_damage = stats["total_damage"] / n if n > 0 else 0
    avg_kills = stats["total_kills"] / n if n > 0 else 0
    avg_deaths = stats["total_deaths"] / n if n > 0 else 0
    avg_assists = stats["total_assists"] / n if n > 0 else 0
    avg_healed = stats["total_healed"] / n if n > 0 else 0
    t1_winrate = stats["t1_wins"] / n * 100 if n > 0 else 0

    # Add main summary document
    summary = f"""DATASET SUMMARY ({n} matches analyzed):
- Average match duration: {int(avg_duration)} seconds ({avg_duration/60:.1f} minutes)
- Average total damage per match: {int(avg_damage)}
- Average total kills per match: {int(avg_kills)}
- Average total deaths per match: {int(avg_deaths)}
- Average total assists per match: {int(avg_assists)}
- Average total healing per match: {int(avg_healed)}
- Team 1 wins: {stats['t1_wins']} ({t1_winrate:.1f}%)
- Team 2 wins: {stats['t2_wins']} ({100-t1_winrate:.1f}%)
- Win rate is balanced due to random team assignment in data processing.
- Heroes available: {', '.join(HERO_NAMES)}"""

    # Add hero statistics document
    hero_stats_lines = ["HERO STATISTICS (pick rates and win rates):"]
    for hero in HERO_NAMES:
        picks = stats["hero_picks"][hero]
        wins = stats["hero_wins"][hero]
        if picks > 0:
            winrate = wins / picks * 100
            pickrate = picks / n * 100
            hero_stats_lines.append(f"- {hero}: {picks} picks ({pickrate:.1f}% pick rate), {wins} wins ({winrate:.1f}% win rate)")
    hero_doc = "\n".join(hero_stats_lines)

    # Add damage statistics document
    damage_doc = f"""DAMAGE STATISTICS:
- Total damage dealt across all {n} matches: {int(stats['total_damage'])}
- Average damage per match (both teams combined): {int(avg_damage)}
- Average damage per team per match: {int(avg_damage/2)}
- Total healing across all matches: {int(stats['total_healed'])}
- Average healing per match: {int(avg_healed)}"""

    # Add kill/death statistics document
    kd_doc = f"""KILL/DEATH STATISTICS:
- Total kills across all {n} matches: {int(stats['total_kills'])}
- Average kills per match (both teams): {int(avg_kills)}
- Total deaths across all matches: {int(stats['total_deaths'])}
- Average deaths per match: {int(avg_deaths)}
- Total assists across all matches: {int(stats['total_assists'])}
- Average assists per match: {int(avg_assists)}
- Average K/D ratio: {avg_kills/avg_deaths:.2f}"""

    # Meta docs for priority in prompts
    meta_docs = [summary, hero_doc, damage_doc, kd_doc]

    # Insert meta docs at beginning for indexing
    for i, meta in enumerate(meta_docs):
        docs.insert(i, meta)

    return docs, meta_docs


def build_index(docs: list[str], collection):
    """Embed documents and store in ChromaDB."""
    print(f"Indexing {len(docs)} documents...")
    for i, doc in enumerate(docs):
        response = ollama.embed(model=EMBED_MODEL, input=doc)
        collection.add(ids=[str(i)], embeddings=[response["embeddings"][0]], documents=[doc])
        if (i + 1) % 500 == 0:
            print(f"  Indexed {i+1} documents...")
    print("Indexing complete.")


def query_rag(question: str, collection, meta_docs: list[str]) -> str:
    """Retrieve relevant docs and generate answer with meta statistics prioritized."""
    q_embed = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[q_embed], n_results=TOP_K)
    retrieved = "\n\n".join(results["documents"][0][:20])  # limit match examples

    # Always include meta statistics at the top
    meta_context = "\n\n".join(meta_docs)

    prompt = f"""You are analyzing Marvel Rivals match data. Use the STATISTICS below to answer aggregate questions (averages, totals, win rates). Only use individual match examples for specific queries.

=== AGGREGATE STATISTICS (use these for questions about averages, totals, win rates) ===
{meta_context}

=== SAMPLE MATCHES (for reference) ===
{retrieved}

Question: {question}

Answer using the aggregate statistics above. Be concise (2-3 sentences):"""

    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# Store meta_docs globally for CLI use
_meta_docs = None


def main():
    global _meta_docs
    client = chromadb.Client()
    collection = client.get_or_create_collection("matches_v3")

    if collection.count() == 0:
        docs, _meta_docs = load_documents(CSV_PATH)
        build_index(docs, collection)
    else:
        # Reload meta docs even if collection exists
        _, _meta_docs = load_documents(CSV_PATH)

    print("\nRAG CLI ready. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            answer = query_rag(question, collection, _meta_docs)
            print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
