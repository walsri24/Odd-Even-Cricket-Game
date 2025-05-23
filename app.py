import streamlit as st
import random
import os
import json
import torch
from collections import defaultdict, Counter
import time
import base64
from game import (
    BALL_COUNT,
    MAX_NUMBER,
    user_bat_comp_ball,
    comp_bat_user_bowl,
    save_batting_sequence,
    update_stats
)
from train_model import train
from ai_model import load_model

# Page configuration
st.set_page_config(
    page_title="Odd-Even Cricket Game",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Custom CSS
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #000000;
        color: #333;
    }

    /* Cricket theme 
    .stApp {
        background-image: linear-gradient(to bottom, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), 
                          url('https://images.pexels.com/photos/3689634/pexels-photo-3689634.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
        background-size: cover;
    }
    */
    
    /* Header styling */
    h1, h2, h3 {
        color: #3E8914;
        font-family: 'Arial', sans-serif;
    }

    /* Card styling */
    .css-1r6slb0, .css-keje6w {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.9);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #3E8914;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Number input buttons */
    .number-btn {
        display: inline-block;
        width: 50px;
        height: 50px;
        margin: 5px;
        border-radius: 50%;
        background-color: #4A90E2;
        color: black;
        font-size: 18px;
        line-height: 50px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .number-btn:hover {
        background-color: #3E8914;
        transform: scale(1.1);
    }

    /* Scoreboard styling */
    .scoreboard {
        background-color: #333;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }

    .score-number {
        font-size: 24px;
        font-weight: bold;
        color: #ffcc00;
    }

    /* Message styling */
    .success-msg {
        color: #3E8914;
        font-weight: bold;
    }

    .danger-msg {
        color: #e74c3c;
        font-weight: bold;
    }

    .info-msg {
        color: #4A90E2;
        font-weight: bold;
    }

    /* Cricket ball animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .cricket-ball {
        width: 40px;
        height: 40px;
        background-color: #c0392b;
        border-radius: 50%;
        display: inline-block;
        animation: spin 2s linear infinite;
        position: relative;
    }

    .cricket-ball:before {
        content: "";
        position: absolute;
        width: 38px;
        height: 19px;
        background-color: #c0392b;
        border: 1px solid #8B0000;
        border-radius: 19px 19px 0 0;
        top: 0;
        left: 0;
    }
    </style>
    """, unsafe_allow_html=True)


load_css()

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'main_menu'
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'batting_team' not in st.session_state:
    st.session_state.batting_team = None
if 'user_score' not in st.session_state:
    st.session_state.user_score = 0
if 'comp_score' not in st.session_state:
    st.session_state.comp_score = 0
if 'ball_count' not in st.session_state:
    st.session_state.ball_count = 0
if 'user_sequence' not in st.session_state:
    st.session_state.user_sequence = []
if 'target' not in st.session_state:
    st.session_state.target = -1
if 'user_choice' not in st.session_state:
    st.session_state.user_choice = 0
if 'comp_choice' not in st.session_state:
    st.session_state.comp_choice = 0
if 'is_out' not in st.session_state:
    st.session_state.is_out = False
if 'message' not in st.session_state:
    st.session_state.message = ""
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = 1
if 'user_batted_first' not in st.session_state:
    st.session_state.user_batted_first = False
if 'innings' not in st.session_state:
    st.session_state.innings = 1
if 'stats' not in st.session_state:
    # Load stats if they exist
    if os.path.exists("data/stats.json"):
        with open("data/stats.json", "r") as f:
            try:
                st.session_state.stats = json.load(f)
            except json.JSONDecodeError:
                st.session_state.stats = {"win": 0, "loss": 0, "tie": 0}
    else:
        st.session_state.stats = {"win": 0, "lose": 0, "tie": 0}


# Helper functions
def reset_game():
    st.session_state.game_state = 'main_menu'
    st.session_state.batting_team = None
    st.session_state.user_score = 0
    st.session_state.comp_score = 0
    st.session_state.ball_count = 0
    st.session_state.user_sequence = []
    st.session_state.target = -1
    st.session_state.user_choice = 0
    st.session_state.comp_choice = 0
    st.session_state.is_out = False
    st.session_state.message = ""
    st.session_state.innings = 1


def set_difficulty(level):
    st.session_state.difficulty = level
    if level == 2:  # Hard mode
        with st.spinner("Loading AI model..."):
            # Ensure the model directory exists
            os.makedirs("data", exist_ok=True)
            # Try loading the model, train if needed
            model_path = "data/model.pth"
            if not os.path.exists(model_path):
                st.info("Training AI model for first use...")
                train()
            st.session_state.ai_model = load_model()
    else:
        st.session_state.ai_model = None
    st.session_state.game_state = 'toss'


def do_toss(choice):
    outcome = random.choice(['H', 'T'])
    if choice == outcome:
        st.session_state.message = "You won the toss!"
        st.session_state.game_state = 'choose_batting'
    else:
        comp_choice = random.choice(["bat", "bowl"])
        st.session_state.message = f"Computer won the toss and chose to {comp_choice} first."
        if comp_choice == "bat":
            st.session_state.batting_team = "computer"
            st.session_state.game_state = 'computer_batting'
        else:
            st.session_state.batting_team = "user"
            st.session_state.game_state = 'user_batting'
        st.session_state.user_batted_first = (st.session_state.batting_team == "user")


def choose_batting(choice):
    if choice == "bat":
        st.session_state.batting_team = "user"
        st.session_state.user_batted_first = True
        st.session_state.game_state = 'user_batting'
    else:
        st.session_state.batting_team = "computer"
        st.session_state.user_batted_first = False
        st.session_state.game_state = 'computer_batting'


def set_user_choice(num):
    st.session_state.user_choice = num


def process_user_batting():
    # Update session state
    st.session_state.user_sequence.append(st.session_state.user_choice)

    # Determine computer's choice
    if st.session_state.difficulty == 3 and st.session_state.ai_model is not None:
        user_seq = st.session_state.user_sequence
        if len(user_seq) > 3 and len(set(user_seq[-3:])) == 1:
            # Pattern detected (e.g., [6,6,6]) — mimic human guess
            comp_choice = user_seq[-1]
        else:
            # Use AI model for prediction
            if len(user_seq) < 3:
                prediction = random.randint(1, MAX_NUMBER)
            else:
                input_seq = torch.tensor([[x - 1 for x in user_seq[-3:]]], dtype=torch.long)
                with torch.no_grad():
                    output = st.session_state.ai_model(input_seq)
                prediction = torch.argmax(output, dim=1).item() + 1

            freq = Counter(user_seq[-3:] if len(user_seq) >= 3 else user_seq)
            most_common = freq.most_common(1)[0][0] if freq else random.randint(1, MAX_NUMBER)
            # Weighted guess between model prediction and frequent move
            comp_choice = random.choices(
                [prediction, most_common],
                weights=[0.9, 0.1]
            )[0]
    else:
        comp_choice = random.randint(1, MAX_NUMBER)

    st.session_state.comp_choice = comp_choice
    st.session_state.ball_count += 1

    # Check if out
    if comp_choice == st.session_state.user_choice:
        st.session_state.is_out = True
        st.session_state.message = "You are OUT!"
        # Save batting sequence
        save_batting_sequence(st.session_state.user_sequence)
        # Check if innings is over
        if st.session_state.innings == 1:
            st.session_state.innings = 2
            st.session_state.target = st.session_state.user_score + 1
            st.session_state.game_state = 'innings_break'
        else:
            st.session_state.game_state = 'game_over'
    else:
        # Add runs
        st.session_state.user_score += st.session_state.user_choice
        st.session_state.message = f"You scored {st.session_state.user_choice} runs!"

        # Check if target achieved
        if st.session_state.target != -1 and st.session_state.user_score >= st.session_state.target:
            st.session_state.game_state = 'game_over'
        # Check if all balls used
        elif st.session_state.ball_count >= BALL_COUNT:
            if st.session_state.innings == 1:
                st.session_state.innings = 2
                st.session_state.target = st.session_state.user_score + 1
                st.session_state.game_state = 'innings_break'
                # Save batting sequence
                save_batting_sequence(st.session_state.user_sequence)
            else:
                st.session_state.game_state = 'game_over'
                # Save batting sequence
                save_batting_sequence(st.session_state.user_sequence)


def process_computer_batting():
    # Update session state for user's bowling choice
    st.session_state.ball_count += 1

    # Determine computer's batting choice
    if st.session_state.difficulty == 1:
        comp_choice = random.randint(1, MAX_NUMBER)
    else:
        # Hard mode logic
        user_bowl_sequence = st.session_state.user_sequence
        if len(user_bowl_sequence) >= 3 and len(set(user_bowl_sequence[-3:])) == 1:
            repeated_number = user_bowl_sequence[-1]
            if random.random() < 0.1:
                comp_choice = repeated_number
            else:
                # Use model prediction
                if len(user_bowl_sequence) < 3:
                    comp_choice = random.randint(1, MAX_NUMBER)
                else:
                    input_seq = torch.tensor([[x - 1 for x in user_bowl_sequence[-3:]]], dtype=torch.long)
                    with torch.no_grad():
                        output = st.session_state.ai_model(input_seq)
                    comp_choice = torch.argmax(output, dim=1).item() + 1
        else:
            # Regular model prediction
            if len(user_bowl_sequence) < 3:
                comp_choice = random.randint(1, MAX_NUMBER)
            else:
                input_seq = torch.tensor([[x - 1 for x in user_bowl_sequence[-3:]]], dtype=torch.long)
                with torch.no_grad():
                    output = st.session_state.ai_model(input_seq)
                comp_choice = torch.argmax(output, dim=1).item() + 1

    st.session_state.comp_choice = comp_choice

    # Check if out
    if comp_choice == st.session_state.user_choice:
        st.session_state.is_out = True
        st.session_state.message = "Computer is OUT!"
        # Check if innings is over
        if st.session_state.innings == 1:
            st.session_state.innings = 2
            st.session_state.target = st.session_state.comp_score + 1
            st.session_state.game_state = 'innings_break'
        else:
            st.session_state.game_state = 'game_over'
    else:
        # Add runs
        st.session_state.comp_score += comp_choice
        st.session_state.message = f"Computer scored {comp_choice} runs!"

        # Check if target achieved
        if st.session_state.target != -1 and st.session_state.comp_score >= st.session_state.target:
            st.session_state.game_state = 'game_over'
        # Check if all balls used
        elif st.session_state.ball_count >= BALL_COUNT:
            if st.session_state.innings == 1:
                st.session_state.innings = 2
                st.session_state.target = st.session_state.comp_score + 1
                st.session_state.game_state = 'innings_break'
            else:
                st.session_state.game_state = 'game_over'


def continue_to_next_innings():
    st.session_state.ball_count = 0
    st.session_state.user_sequence = []
    st.session_state.is_out = False
    st.session_state.message = ""

    if st.session_state.batting_team == "user":
        st.session_state.batting_team = "computer"
        st.session_state.game_state = 'computer_batting'
    else:
        st.session_state.batting_team = "user"
        st.session_state.game_state = 'user_batting'


def determine_winner():
    if st.session_state.user_batted_first:
        if st.session_state.user_score > st.session_state.comp_score:
            result = "win"
            message = "🎉 YOU WIN!"
        elif st.session_state.user_score < st.session_state.comp_score:
            result = "lose"
            message = "😞 YOU LOSE!"
        else:
            result = "tie"
            message = "🤝 MATCH TIED!"
    else:
        if st.session_state.user_score > st.session_state.comp_score:
            result = "win"
            message = "🎉 YOU WIN!"
        elif st.session_state.user_score < st.session_state.comp_score:
            result = "lose"
            message = "😞 YOU LOSE!"
        else:
            result = "tie"
            message = "🤝 MATCH TIED!"

    update_stats(result)
    return message


# Main menu
if st.session_state.game_state == 'main_menu':
    st.title("🏏 Odd-Even Cricket Game")

    st.markdown("""
    ### Welcome to the Odd-Even Cricket Game!

    This is a virtual version of the popular hand cricket game where you pick numbers between 1-9.
    If the bowler guesses the same number as you, you're out! Otherwise, you score the runs you picked.

    **How to play:**
    1. Choose difficulty level
    2. Win the toss to decide batting or bowling
    3. Pick numbers when batting or bowling
    4. Score more runs than your opponent to win!

    Ready to play? Choose your difficulty level:
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Easy Mode", use_container_width=True):
            set_difficulty(1)
    with col2:
        if st.button("Hard Mode (AI)", use_container_width=True):
            set_difficulty(3)

    # Show stats if available
    if st.session_state.stats:
        st.markdown("### Your Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Wins", st.session_state.stats["wins"])
        col2.metric("Losses", st.session_state.stats["losses"])
        col3.metric("Ties", st.session_state.stats["ties"])

# Toss screen
elif st.session_state.game_state == 'toss':
    st.title("🏏 Coin Toss")
    st.markdown("### Call Heads or Tails")

    # Add coin animation
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div class="cricket-ball"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Heads", use_container_width=True):
            do_toss('H')
    with col2:
        if st.button("Tails", use_container_width=True):
            do_toss('T')

# Choose batting or bowling
elif st.session_state.game_state == 'choose_batting':
    st.title("🏏 You won the toss!")
    st.markdown("### What would you like to do?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Bat first", use_container_width=True):
            choose_batting("bat")
    with col2:
        if st.button("Bowl first", use_container_width=True):
            choose_batting("bowl")

# User batting
elif st.session_state.game_state == 'user_batting':
    innings_text = "First" if st.session_state.innings == 1 else "Second"
    st.title(f"🏏 {innings_text} Innings: Your Batting")

    # Scoreboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Your Score: {st.session_state.user_score}")
    with col2:
        st.markdown(f"### Ball: {st.session_state.ball_count + 1}/{BALL_COUNT}")
    with col3:
        if st.session_state.target != -1:
            st.markdown(f"### Target: {st.session_state.target}")

    # Show message from previous ball
    if st.session_state.message:
        if "OUT" in st.session_state.message:
            st.markdown(f'<p class="danger-msg">{st.session_state.message}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="success-msg">{st.session_state.message}</p>', unsafe_allow_html=True)

    # Show computer's choice if a ball was played
    if st.session_state.comp_choice > 0:
        st.markdown(f"Computer bowled: **{st.session_state.comp_choice}**")

    # Number selection buttons
    st.markdown("### Select your batting number:")
    cols = st.columns(9)
    for i in range(9):
        num = i + 1
        with cols[i]:
            if st.button(f"{num}", key=f"bat_{num}", use_container_width=True):
                set_user_choice(num)
                process_user_batting()
                st.rerun()

    # Ball animation if waiting for input
    if not st.session_state.is_out:
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <div style="font-style: italic; color: #666;">Waiting for your choice...</div>
        </div>
        """, unsafe_allow_html=True)

# Computer batting
elif st.session_state.game_state == 'computer_batting':
    innings_text = "First" if st.session_state.innings == 1 else "Second"
    st.title(f"🏏 {innings_text} Innings: Computer Batting")

    # Scoreboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Computer Score: {st.session_state.comp_score}")
    with col2:
        st.markdown(f"### Ball: {st.session_state.ball_count + 1}/{BALL_COUNT}")
    with col3:
        if st.session_state.target != -1:
            st.markdown(f"### Target: {st.session_state.target}")

    # Show message from previous ball
    if st.session_state.message:
        if "OUT" in st.session_state.message:
            st.markdown(f'<p class="success-msg">{st.session_state.message}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="danger-msg">{st.session_state.message}</p>', unsafe_allow_html=True)

    # Show choices if a ball was played
    if st.session_state.comp_choice > 0:
        st.markdown(f"Computer batted: **{st.session_state.comp_choice}**")
        st.markdown(f"You bowled: **{st.session_state.user_choice}**")

    # Number selection buttons for bowling
    st.markdown("### Select your bowling number:")
    cols = st.columns(9)
    for i in range(9):
        num = i + 1
        with cols[i]:
            if st.button(f"{num}", key=f"bowl_{num}", use_container_width=True):
                set_user_choice(num)
                st.session_state.user_sequence.append(num)
                process_computer_batting()
                st.rerun()

    # Ball animation if waiting for input
    if not st.session_state.is_out:
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <div style="font-style: italic; color: #666;">Waiting for your choice...</div>
        </div>
        """, unsafe_allow_html=True)

# Innings break
elif st.session_state.game_state == 'innings_break':
    st.title("🏏 Innings Break")

    # Show first innings summary
    if st.session_state.batting_team == "user":
        st.markdown(f"### Your Innings: {st.session_state.user_score} runs")
        st.markdown(f"### Target for Computer: {st.session_state.target} runs")
    else:
        st.markdown(f"### Computer Innings: {st.session_state.comp_score} runs")
        st.markdown(f"### Target for You: {st.session_state.target} runs")

    # Continue button
    if st.button("Start Second Innings", use_container_width=True):
        continue_to_next_innings()
        st.rerun()

# Game over
elif st.session_state.game_state == 'game_over':
    st.title("🏏 Match Result")

    # Show final scores
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Your Score: {st.session_state.user_score}")
    with col2:
        st.markdown(f"### Computer Score: {st.session_state.comp_score}")

    # Show result
    result_message = determine_winner()
    st.markdown(f"## {result_message}", unsafe_allow_html=True)

    # Show current stats
    st.markdown("### Your Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Wins", st.session_state.stats["wins"])
    col2.metric("Losses", st.session_state.stats["losses"])
    col3.metric("Ties", st.session_state.stats["ties"])

    # Play again button
    if st.button("Play Again", use_container_width=True):
        reset_game()
        st.rerun()