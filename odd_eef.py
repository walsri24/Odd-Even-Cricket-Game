import random

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
        
def comp_bat_user_bowl(sc):
    ask = -1
    score = 0
    while True:
        ask = int(input("Enter number between 1 & 6:    "))
        while ask > 6 or ask < 1:
            ask = int(input("Enter number between 1 & 6:    "))

        comp_run = random.randint(1,6)
        
        if comp_run == ask:
            print("Computer is out")
            print(f"Final Computer Score:   {score}")
            break
        else:
            score += comp_run
            if score >= sc and sc != -1:
                print(f"Computer's number:  {comp_run}")
                print(f"Computer's score:   {score}")                
                break

            print(f"Computer's number:  {comp_run}")
            print(f"Computer's score:   {score}")
            

    return score


def user_bat_comp_ball(sc):
    ask = -1
    score = 0
    while True:
        ask = int(input("Enter number between 1 & 6:    "))
        while ask > 6 or ask < 1:
            ask = int(input("Enter number between 1 & 6:    "))
            
        comp_ball = random.randint(1,6)
        if comp_ball == ask:
            # print(f"Computer's number {comp_ball}")
            print("You are out")
            print(f"Final User Score: {score}")
            break
        else:
            score += ask
            if score >= sc and sc != -1:
                # print(f"Computer's number {comp_ball}")
                print(f"Your Score:     {score}")
                break
            # print(f"Computer's number {comp_ball}")
            print(f"Your Score:     {score}")
            
    return score
    
        


def odd_eef():
    winner, res = toss()
    # winner="user"
    # res="bowl"
    # print(winner, res)
    if (winner == "user" and res == "bowl") or (winner == "computer" and res == "bat"):
        print("First Innings")
        print("Computer Batting And User Bowling")
        f_score = comp_bat_user_bowl(-1)
        print()
        print("Second Innings")
        print("User Batting And Computer Bowling")
        s_score = user_bat_comp_ball(f_score+1)
        print()
        if f_score > s_score:
            print("YOU LOSE")
        elif f_score < s_score:
            print("YOU WIN")
    
    if (winner == "user" and res == "bat") or (winner == "computer" and res == "bowl"):
        print("First Innings")
        print("User Batting And Computer Bowling")
        f_score = user_bat_comp_ball(-1)
        print()
        print("Second Innings")
        print("Computer Batting And User Bowling")
        s_score = comp_bat_user_bowl(f_score+1)
        print()
        if f_score < s_score:
            print("YOU LOSE")
        elif f_score > s_score:
            print("YOU WIN")



# toss()
# comp_bat_user_bowl()
# user_bat_comp_ball()
odd_eef()