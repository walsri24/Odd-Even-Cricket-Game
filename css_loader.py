import streamlit as st

def load_css():
    css = """
    <style>
        /* Center everything */
        .main .block-container {
            max-width: 700px;
            padding-top: 2rem;
        }

        /* Custom buttons */
        button[kind="primary"] {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }

        button[kind="primary"]:hover {
            background-color: #45a049;
        }

        /* Title and markdown styling */
        h1, h2, h3 {
            text-align: center;
            color: #2e7d32;
        }

        .stMarkdown {
            font-size: 16px;
            line-height: 1.6;
        }

        /* Metric customization */
        .stMetric {
            text-align: center;
        }

        /* Spinner styling */
        .stSpinner > div {
            text-align: center;
            color: #2e7d32;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
