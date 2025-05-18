import random
from collections import defaultdict

def toss():
    ask = input("Heads or Tails? (H/T): ").upper()
    l = ['H', 'T']
    outcome = random.choice(l)
    des = ["bat", "bowl"]
    winner = ""
    res = ""
    if ask == outcome:
        print("You have won the toss")
        print("Bat (Press 1)")
        print("Bowl (Press 2)")
        ask_2 = int(input("Choose:- "))
        winner = "user"
        if ask_2 == 1:
            res = des[0]
        else:
            res = des[1]
    else:
        out = random.choice(des)
        print(f"Computer has won the toss and chose to {out} first")
        winner = "computer"
        res = out
    return winner, res

# AI memory
user_bowl_memory = defaultdict(int)
user_bat_memory = defaultdict(int)

def comp_bat_user_bowl(sc):
    score = 0
    while True:
        ask = int(input("Enter number between 1 & 6:    "))
        while ask > 6 or ask < 1:
            ask = int(input("Enter number between 1 & 6:    "))
        
        user_bowl_memory[ask] += 1

        # Computer avoids user's common choices
        weights = [1 / (user_bowl_memory[i] + 1) for i in range(1, 7)]
        comp_run = random.choices(range(1, 7), weights=weights)[0]

        if comp_run == ask:
            print("Computer is out")
            print(f"Final Computer Score:   {score}")
            break
        else:
            score += comp_run
            print(f"Computer's number:  {comp_run}")
            print(f"Computer's score:   {score}")
            if score >= sc and sc != -1:
                break

    return score

def user_bat_comp_ball(sc):
    score = 0
    while True:
        ask = int(input("Enter number between 1 & 6:    "))
        while ask > 6 or ask < 1:
            ask = int(input("Enter number between 1 & 6:    "))
        
        user_bat_memory[ask] += 1

        # Computer bowls user's common numbers more often to try to get them out
        weights = [(user_bat_memory[i] + 1) for i in range(1, 7)]
        comp_ball = random.choices(range(1, 7), weights=weights)[0]

        if comp_ball == ask:
            print("You are out")
            print(f"Final User Score: {score}")
            break
        else:
            score += ask
            print(f"Your Score:     {score}")
            if score >= sc and sc != -1:
                break

    return score

def odd_eef():
    winner, res = toss()

    if (winner == "user" and res == "bowl") or (winner == "computer" and res == "bat"):
        print("\nFirst Innings")
        print("Computer Batting And User Bowling")
        f_score = comp_bat_user_bowl(-1)
        print("\nSecond Innings")
        print("User Batting And Computer Bowling")
        s_score = user_bat_comp_ball(f_score + 1)
        print()
        if f_score > s_score:
            print("YOU LOSE")
        elif f_score < s_score:
            print("YOU WIN")
        else:
            print("MATCH TIED")

    if (winner == "user" and res == "bat") or (winner == "computer" and res == "bowl"):
        print("\nFirst Innings")
        print("User Batting And Computer Bowling")
        f_score = user_bat_comp_ball(-1)
        print("\nSecond Innings")
        print("Computer Batting And User Bowling")
        s_score = comp_bat_user_bowl(f_score + 1)
        print()
        if f_score < s_score:
            print("YOU LOSE")
        elif f_score > s_score:
            print("YOU WIN")
        else:
            print("MATCH TIED")

# Start game
odd_eef()
