import json
import os

def save_sequence(seq, filename="data/bat_sequences.json"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(filename):
        try:
            with open(filename) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append(seq)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_sequences(filename="data/bat_sequences.json"):
    if os.path.exists(filename):
        with open(filename) as f:
            try:
                sequences = json.load(f)  # use json.load, not json.loads
                return sequences
            except Exception as e:
                return {"error": str(e)}
    return []


def update_stats(result, filename="data/stats.json"):
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Initialize stats from session state or file
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                stats = json.load(f)
        except json.JSONDecodeError:
            stats = {"games": 0, "wins": 0, "losses": 0, "ties": 0}
    else:
        stats = {"games": 0, "wins": 0, "losses": 0, "ties": 0}

    # Update stats
    stats["games"] += 1
    if result == "win":
        stats["wins"] += 1
    elif result == "lose":
        stats["losses"] += 1
    else:
        stats["ties"] += 1

    # Save to file
    with open(filename, "w") as f:
        json.dump(stats, f, indent=4)


def get_statistics(file_path='data/stats.json'):
    if not os.path.exists(file_path):
        return "📉 No stats available yet. Play a game to generate your stats!"

    with open(file_path, 'r') as f:
        stats = json.load(f)

    games = stats.get("games", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    ties = stats.get("ties", 0)
    win_rate = (wins / games * 100) if games > 0 else 0

    return (
        f"🏏 *Your Game Stats*\n\n"
        f"🎮 Total Games: *{games}*\n"
        f"✅ Wins: *{wins}*\n"
        f"❌ Losses: *{losses}*\n"
        f"🤝 Ties: *{ties}*\n"
        f"📊 Win Rate: *{win_rate:.2f}%*"
    )
