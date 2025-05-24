import logging
import os
import random
import torch
from collections import defaultdict, Counter
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler, filters,
                          ContextTypes, ConversationHandler)

from ai_model import load_model
from train_model import train
from utils import get_statistics, update_stats
from dotenv import load_dotenv

load_dotenv()
# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# States for ConversationHandler
DIFFICULTY, TOSS_CHOICE, BAT_OR_BOWL, GAMEPLAY = range(4)

# Constants
BALL_COUNT = 18
MAX_NUMBER = 9

# User game sessions
game_sessions = {}

# Helper dataclass
class GameSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.difficulty = 1
        self.model = None
        self.toss_result = None
        self.user_batted_first = False
        self.user_score = 0
        self.comp_score = 0
        self.user_sequence = []
        self.ball_count = 0
        self.target = -1
        self.is_batting = True
        self.awaiting_input = False

# Start command
async def start_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    game_sessions[user_id] = GameSession(user_id)

    # Train model in background
    from threading import Thread
    Thread(target=train).start()

    await update.message.reply_text(
        "Choose difficulty:",
        reply_markup=ReplyKeyboardMarkup([
            ["Easy"],
            ["Hard"]
        ], one_time_keyboard=True, resize_keyboard=True)
    )
    return DIFFICULTY

async def select_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    diff_text = update.message.text
    session = game_sessions[user_id]
    session.difficulty = 1 if diff_text.lower() == "easy" else 2
    if session.difficulty == 2:
        session.model = load_model()

    await update.message.reply_text(
        "Toss time! Choose Heads or Tails:",
        reply_markup=ReplyKeyboardMarkup([
            ["Heads"],
            ["Tails"]
        ], one_time_keyboard=True, resize_keyboard=True)
    )
    return TOSS_CHOICE

async def toss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = game_sessions[user_id]
    user_choice = update.message.text[0].upper()
    result = random.choice(['H', 'T'])
    des = ["bat", "bowl"]

    if user_choice == result:
        await update.message.reply_text(
            "You won the toss! Choose:",
            reply_markup=ReplyKeyboardMarkup([
                ["Bat"],
                ["Bowl"]
            ], one_time_keyboard=True, resize_keyboard=True)
        )
        return BAT_OR_BOWL
    else:
        comp_choice = random.choice(des)
        session.is_batting = comp_choice == "bowl"  # if comp bowls, user bats
        session.user_batted_first = session.is_batting
        await update.message.reply_text(f"Computer won the toss and chose to {comp_choice} first.")
        return await start_innings(update, context)

async def choose_bat_or_bowl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = game_sessions[user_id]
    user_choice = update.message.text.lower()
    session.is_batting = user_choice == "bat"
    session.user_batted_first = session.is_batting
    return await start_innings(update, context)

async def start_innings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = game_sessions[user_id]
    session.ball_count = 0
    session.user_sequence = []
    session.awaiting_input = True

    role = "Batting" if session.is_batting else "Bowling"
    await update.message.reply_text(
        f"You are now {role}. Send a number (1-{MAX_NUMBER}) per ball."
    )
    return GAMEPLAY

async def handle_gameplay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = game_sessions[user_id]

    if not session.awaiting_input:
        return GAMEPLAY

    try:
        user_input = int(update.message.text)
        if not (1 <= user_input <= MAX_NUMBER):
            raise ValueError
    except ValueError:
        await update.message.reply_text(f"Enter a number between 1 and {MAX_NUMBER}.")
        return GAMEPLAY

    session.user_sequence.append(user_input)
    session.ball_count += 1

    # AI or random prediction
    if session.difficulty == 1 or session.model is None:
        comp_choice = random.randint(1, MAX_NUMBER)
    else:
        if len(session.user_sequence) < 3:
            comp_choice = random.randint(1, MAX_NUMBER)
        else:
            input_seq = torch.tensor([[x - 1 for x in session.user_sequence[-3:]]], dtype=torch.long)
            with torch.no_grad():
                output = session.model(input_seq)
            comp_choice = torch.argmax(output, dim=1).item() + 1

    ball_msg = f"Ball {session.ball_count}/{BALL_COUNT}"
    await update.message.reply_text(f"Ball {session.ball_count}/{BALL_COUNT}")

    if session.is_batting:
        if comp_choice == user_input:
            await update.message.reply_text(f"You are OUT! Final Score: {session.user_score}\n{ball_msg}")
            if session.user_batted_first:
                session.target = session.user_score + 1
                session.is_batting = False
                return await start_innings(update, context)
            else:
                return await match_result(update, context)

        else:
            session.user_score += user_input
            await update.message.reply_text(f"You scored {user_input}. Total: {session.user_score}\n{ball_msg}")

            # If chasing
            if not session.user_batted_first and session.user_score >= session.target:
                await update.message.reply_text(f"You reached the target! 🎉\n{ball_msg}")
                return await match_result(update, context)

    else:  # Bowling
        if comp_choice == user_input:
            await update.message.reply_text(f"Computer is OUT! Final Score: {session.comp_score}\n{ball_msg}")
            if session.user_batted_first:
                return await match_result(update, context)
            else:
                session.is_batting = True
                session.user_batted_first = True  # Now the user bats
                session.target = session.comp_score + 1
                return await start_innings(update, context)
        else:
            session.comp_score += comp_choice
            await update.message.reply_text(f"Computer scored {comp_choice}. Total: {session.comp_score}\n{ball_msg}")

            if session.user_batted_first and session.comp_score >= session.target:
                await update.message.reply_text(f"Computer reached the target! 😞\n{ball_msg}")
                return await match_result(update, context)

    if session.ball_count >= BALL_COUNT:
        if session.is_batting:
            await update.message.reply_text(f"Innings over! Your Score: {session.user_score}\n{ball_msg}")
            session.target = session.user_score + 1
            session.is_batting = False
            return await start_innings(update, context)
        else:
            await update.message.reply_text(f"Innings over! Computer Score: {session.comp_score}\n{ball_msg}")
            return await match_result(update, context)

    return GAMEPLAY


async def match_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = game_sessions[update.effective_user.id]
    msg = ""

    if session.user_batted_first:
        # User batted first
        if session.user_score > session.comp_score:
            msg = "🎉 YOU WIN!"
        elif session.user_score < session.comp_score:
            msg = "😞 YOU LOSE!"
        else:
            msg = "🤝 MATCH TIED!"
    else:
        # User batted second
        if session.user_score >= session.target:
            msg = "🎉 YOU WIN!"
        elif session.ball_count >= BALL_COUNT or session.user_score < session.target:
            msg = "😞 YOU LOSE!"
        else:
            msg = "🤝 MATCH TIED!"  # unlikely fallback

    await update.message.reply_text(f"Game Over. {msg}", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Game cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# Get stats dummy (replace with real file read later)
async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_statistics()
    await update.message.reply_text(text, parse_mode="Markdown")

# Help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
/start_game - Start a new game
/get_stats - Show your stats
/help - Show this message
/cancel - Cancel current game
""")

# Entry point
if __name__ == "__main__":
    app = ApplicationBuilder().token(os.getenv("BOT_TOKEN")).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start_game", start_game)],
        states={
            DIFFICULTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_difficulty)],
            TOSS_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, toss)],
            BAT_OR_BOWL: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_bat_or_bowl)],
            GAMEPLAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_gameplay)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("get_stats", get_stats))
    app.add_handler(CommandHandler("help", help_command))

    app.run_polling()
