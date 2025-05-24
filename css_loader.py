import streamlit as st

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #000000;
        color: #333;
    }

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
