import random
from collections import defaultdict
from utils import save_sequence, update_stats
from ai_model import load_model
import torch
import os
import json

# Global memory
user_bat_memory = defaultdict(int)
user_bowl_memory = defaultdict(int)
user_bat_sequence = []
user_bowl_sequence = []


def save_batting_sequence(sequence):
    os.makedirs("data", exist_ok=True)
    file_path = "data/bat_sequences.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(sequence)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


# Ask difficulty
def select_difficulty():
    print("\nChoose Difficulty:")
    print("1. Easy")
    print("2. Medium (AI adapts to your habits)")
    print("3. Hard (AI learns patterns over time)")
    while True:
        try:
            level = int(input("Enter 1/2/3: "))
            if level in [1, 2, 3]:
                return level
        except:
            pass
        print("Invalid choice.")

# Toss logic
def toss():
    ask = input("Heads or Tails? (H/T): ").upper()
    outcome = random.choice(['H', 'T'])
    des = ["bat", "bowl"]

    if ask == outcome:
        print("You won the toss!")
        print("1. Bat")
        print("2. Bowl")
        choice = int(input("Choose: "))
        return "user", des[choice - 1]
    else:
        comp_choice = random.choice(des)
        print(f"Computer won the toss and chose to {comp_choice} first.")
        return "computer", comp_choice

# Predict next move using LSTM
def ai_predict_next(seq, model):
    if len(seq) < 5:
        return random.randint(1, 6)
    input_seq = torch.tensor([seq[-5:]], dtype=torch.long)
    with torch.no_grad():
        output = model(input_seq)
    return torch.argmax(output, dim=1).item() + 1

# Computer bats
def comp_bat_user_bowl(target, difficulty=2, model=None):
    score = 0
    ball_count = 0
    print("\n--- Computer Batting (2 Overs Max) ---")

    while ball_count < 12:
        try:
            user_input = int(input(f"Ball {ball_count+1}/12 - Enter number (1-6): "))
            if not (1 <= user_input <= 6): continue
        except:
            continue

        user_bowl_memory[user_input] += 1
        user_bowl_sequence.append(user_input)

        if difficulty == 1:
            comp_choice = random.randint(1, 6)
        elif difficulty == 2:
            weights = [1 / (user_bowl_memory[i] + 1) for i in range(1, 7)]
            comp_choice = random.choices(range(1, 7), weights=weights)[0]
        else:
            comp_choice = ai_predict_next(user_bowl_sequence, model)

        print(f"Computer chose: {comp_choice}")
        ball_count += 1

        if comp_choice == user_input:
            print(f"OUT! Final Computer Score: {score}")
            break
        else:
            score += comp_choice
            print(f"Computer Score: {score}")
            if target != -1 and score >= target:
                break

    if ball_count == 12:
        print("Innings Over (12 balls)")
        print(f"Final Computer Score: {score}")

    return score

# User bats
def user_bat_comp_ball(target, model=None, difficulty=1):
    score = 0
    user_bat_sequence = []
    ball_count = 0
    print("\n--- You Batting (2 Overs Max) ---")

    while ball_count < 12:
        try:
            ask = int(input(f"Ball {ball_count+1}/12 - Enter number (1-6): "))
        except ValueError:
            continue
        while ask < 1 or ask > 6:
            ask = int(input("Enter number (1-6): "))

        user_bat_sequence.append(ask)

        if difficulty == 3 and model is not None:
            comp_choice = ai_predict_next(user_bat_sequence, model)
        else:
            comp_choice = random.randint(1, 6)

        print(f"Computer bowled: {comp_choice}")
        ball_count += 1

        if comp_choice == ask:
            print("You are OUT!")
            print(f"Your Final Score: {score}")
            break
        else:
            score += ask
            print(f"Your Score: {score}")
            if target != -1 and score >= target:
                break

    if ball_count == 12:
        print("Innings Over (12 balls)")
        print(f"Your Final Score: {score}")

    if difficulty in [2, 3]:
        save_batting_sequence(user_bat_sequence)

    return score

# Main game
def play_game():
    difficulty = select_difficulty()
    if difficulty == 3:
        ai_model = load_model()
    else:
        ai_model = None

    winner, res = toss()
    user_batted_first = False

    if (winner == "user" and res == "bat") or (winner == "computer" and res == "bowl"):
        user_batted_first = True
        print("\nFirst Innings: You Bat")
        user_score = user_bat_comp_ball(-1, ai_model, difficulty)
        print(f"Target for Computer is {user_score + 1}")

        print("\nSecond Innings: Computer Bats")
        comp_score = comp_bat_user_bowl(
            target=user_score + 1,
            difficulty=difficulty,
            model=ai_model
        )

    else:
        user_batted_first = False
        print("\nFirst Innings: Computer Bats")
        comp_score = comp_bat_user_bowl(
            target=-1,
            difficulty=difficulty,
            model=ai_model
        )

        print(f"Target for You is {comp_score + 1}")
        print("\nSecond Innings: You Bat")
        user_score = user_bat_comp_ball(comp_score + 1, ai_model, difficulty)

    print("\n--- Match Result ---")
    if user_batted_first:
        if user_score > comp_score:
            print("🎉 YOU WIN!")
            update_stats("win")
        elif user_score < comp_score:
            print("😞 YOU LOSE!")
            update_stats("lose")
        else:
            print("🤝 MATCH TIED!")
            update_stats("tie")
    else:
        if user_score > comp_score:
            print("🎉 YOU WIN!")
            update_stats("win")
        elif user_score < comp_score:
            print("😞 YOU LOSE!")
            update_stats("lose")
        else:
            print("🤝 MATCH TIED!")
            update_stats("tie")

if __name__ == "__main__":
    play_game()
