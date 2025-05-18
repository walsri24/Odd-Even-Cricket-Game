import random
ROWS = 3

def generate_grid():
    matrix = []
    for i in range(ROWS):
        row = list(eval(input(f"Enter Row {i+1}:- ")))
        matrix.append(row)
    return matrix


def get_bingo_ball(used_balls): 
    all_balls = [i for i in range(1, 26)] 
    available_balls = set(all_balls) - used_balls 
    if available_balls: 
        # ball = random.choice(list(available_balls)) 
        ball = int(input("Enter Card Number:- "))
        used_balls.add(ball) 
        return ball 
    else: 
        return None


def check_bingo(card): 
    rows = card 
    cols = [[card[j][i] for j in range(ROWS)] for i in range(ROWS)] 
    diags = [[card[i][i] for i in range(ROWS)], [card[i][ROWS-1-i] for i in range(ROWS)]] 
    lines = rows + cols + diags 
    for line in lines: 
        if len(set(line)) == 1 and line[0] != 0: 
            return True
    return False



def play_bingo():
    card = generate_grid()   
    print("Bingo Card")
    for i in card:
        print(' '.join([str(n).rjust(2) for n in i]))
    print()
    

    # used_card = set()
    while True:

        if check_bingo(card):
            print("You Win")
            break 
    
    
        ball = int(input("Enter Card Number:- "))

        # if ball in None:
        #     print("All Bingo cards have been drawn. The Game is a tie")
        #     break
        
        print("New Bingo Ball", ball)
        
        
        # input("Press Enter to draw the next Bingo ball..")
        
        for i in range(ROWS):
            for j in range(ROWS):
                if card[i][j] == ball:
                    card[i][j] = "X"
                
        print("Bingo Card: ")
        for row in card:
            print(' '.join([str(n).rjust(2) if isinstance(n, int) else n.rjust(2) for n in row]))
            
        print()
        
play_bingo()
    
    
