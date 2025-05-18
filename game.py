import random
from collections import defaultdict
from utils import save_sequence, update_stats
from ai_model import load_model
import torch

# Global memory
user_bat_memory = defaultdict(int)
user_bowl_memory = defaultdict(int)
user_bat_sequence = []
user_bowl_sequence = []

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
def comp_bat_user_bowl(target, difficulty, model=None):
    score = 0
    print("\n--- Computer Batting ---")
    while True:
        try:
            user_input = int(input("Enter number (1-6): "))
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
        if comp_choice == user_input:
            print(f"OUT! Final Computer Score: {score}")
            break
        else:
            score += comp_choice
            print(f"Computer Score: {score}")
            if target != -1 and score >= target:
                break

    return score

# User bats
def user_bat_comp_ball(target, difficulty, model=None):
    score = 0
    print("\n--- You Batting ---")
    while True:
        try:
            user_input = int(input("Enter number (1-6): "))
            if not (1 <= user_input <= 6): continue
        except:
            continue

        user_bat_memory[user_input] += 1
        user_bat_sequence.append(user_input)

        if difficulty == 1:
            comp_choice = random.randint(1, 6)
        elif difficulty == 2:
            weights = [(user_bat_memory[i] + 1) for i in range(1, 7)]
            comp_choice = random.choices(range(1, 7), weights=weights)[0]
        else:
            comp_choice = ai_predict_next(user_bat_sequence, model)

        print(f"Computer bowled: {comp_choice}")
        if comp_choice == user_input:
            print(f"OUT! Final Your Score: {score}")
            break
        else:
            score += user_input
            print(f"Your Score: {score}")
            if target != -1 and score >= target:
                break

    return score

# Main game
def play_game():
    difficulty = select_difficulty()
    model = load_model() if difficulty == 3 else None

    winner, decision = toss()

    if (winner == "user" and decision == "bowl") or (winner == "computer" and decision == "bat"):
        print("\nFirst Innings: Computer Bats")
        target = comp_bat_user_bowl(-1, difficulty, model)
        print("\nSecond Innings: You Bat")
        score = user_bat_comp_ball(target + 1, difficulty, model)
    else:
        print("\nFirst Innings: You Bat")
        target = user_bat_comp_ball(-1, difficulty, model)
        print("\nSecond Innings: Computer Bats")
        score = comp_bat_user_bowl(target + 1, difficulty, model)

    # Result
    print()
    if score > target:
        print("🎉 YOU WIN!")
        update_stats("win")
    elif score < target:
        print("😞 YOU LOSE!")
        update_stats("lose")
    else:
        print("🤝 MATCH TIED!")
        update_stats("tie")

    # Save user pattern
    if difficulty == 3:
        save_sequence(user_bat_sequence)

if __name__ == "__main__":
    play_game()
