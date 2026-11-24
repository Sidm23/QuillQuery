
import streamlit as st
import pandas as pd
import altair as alt

from pathlib import Path

from app import create_vector_store, answer_question, generate_mcqs
from ingestion import load_document

from database import (
    initialize_database,
    save_quiz_result,
    get_quiz_history,
    get_performance,
    get_weak_topics
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="QuillQuery | Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_database()


# ==================================================
# 2. SESSION STATE
# ==================================================

defaults = {
    "vector_store": None,
    "bm25_retriever": None,
    "messages": [],
    "uploaded_files": [],
    "quiz_questions": [],
    "quiz_submitted": False,
    "quiz_answers": {},
    "quiz_score": 0,
    "quiz_topic": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 3. THEME COLORS — DARK ONLY
# ==================================================

dark = True

bg = "#090A0C"
sidebar = "#0D0F12"
surface = "rgba(255,255,255,0.035)"
surface_hover = "rgba(255,255,255,0.065)"
border = "rgba(220,220,220,0.13)"
text = "#F2F2F0"
muted = "#A1A1A1"
accent = "#B8B8B8"
accent2 = "#D0D0D0"
input_bg = "#141517"
uploader_bg = "#131416"
alert_bg = "#252629"
alert_text = "#EEEEEE"
chart_bg = "#111214"
chart_text = "#E4E4E4"
chart_grid = "#303236"
progress_track = "#34363A"
table_header = "#242629"
table_row = "#131416"
table_alt = "#191B1E"
shadow = "rgba(0,0,0,0.38)"



# ==================================================
# 4. CUSTOM CSS
# ==================================================

st.markdown(
    f"""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    :root {{
        --bg: {bg};
        --sidebar: {sidebar};
        --surface: {surface};
        --surface-hover: {surface_hover};
        --border: {border};
        --text: {text};
        --muted: {muted};
        --accent: {accent};
        --accent2: {accent2};
        --input-bg: {input_bg};
        --uploader-bg: {uploader_bg};
        --alert-bg: {alert_bg};
        --alert-text: {alert_text};
        --chart-bg: {chart_bg};
        --chart-text: {chart_text};
        --chart-grid: {chart_grid};
        --progress-track: {progress_track};
        --shadow: {shadow};
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(125, 90, 230, 0.13),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(70, 180, 230, 0.10),
                transparent 28%
            ),
            var(--bg) !important;
        color: var(--text) !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }}

    /* SIDEBAR */

    [data-testid="stSidebar"] {{
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}

    /* TYPOGRAPHY */

    h1, h2, h3, h4 {{
        color: var(--text) !important;
        font-weight: 700 !important;
        letter-spacing: -0.6px;
    }}

    p, li, label {{
        color: var(--text);
    }}

    [data-testid="stCaptionContainer"] p {{
        color: var(--muted) !important;
    }}

    /* HERO */

    .hero {{
        padding: 30px;
        border-radius: 24px;
        border: 1px solid var(--border);
        background:
            linear-gradient(
                120deg,
                rgba(133, 91, 225, 0.20),
                rgba(58, 157, 214, 0.10)
            ),
            var(--surface);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 12px 35px var(--shadow);
        margin-bottom: 24px;
    }}

    .hero-label {{
        color: var(--accent2);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .hero-title {{
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: var(--text);
        margin-top: 10px;
        line-height: 1.2;
    }}

    .hero-description {{
        color: var(--muted);
        font-size: 15px;
        margin-top: 10px;
        line-height: 1.7;
    }}

    /* BRAND */

    .brand {{
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--text);
    }}

    .brand span {{
        color: var(--accent);
    }}

    .sidebar-note {{
        color: var(--muted);
        font-size: 12px;
        line-height: 1.7;
    }}

    /* METRICS */

    [data-testid="stMetric"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        backdrop-filter: blur(18px);
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {{
        color: var(--text) !important;
    }}

    /* INPUTS */

    .stTextInput input,
    .stTextArea textarea {{
        background: var(--input-bg) !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }}

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
    }}

    /* SELECTBOX */

    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }}

    [data-testid="stSelectbox"] span {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    [data-baseweb="popover"],
    [data-baseweb="menu"] {{
        background: var(--input-bg) !important;
    }}

    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {{
        color: var(--text) !important;
    }}

    /* UPLOADER */

    [data-testid="stFileUploader"] {{
        background: var(--uploader-bg) !important;
        border: 1px dashed var(--accent) !important;
        border-radius: 16px !important;
        padding: 12px !important;
    }}

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div {{
        background: var(--uploader-bg) !important;
        border: none !important;
    }}

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }}

    [data-testid="stFileUploader"] button {{
        background: var(--surface-hover) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }}

    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFile"] > div > div {{
        background: var(--surface-hover) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }}

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] p {{
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        opacity: 1 !important;
    }}

    /* ALERTS */

    [data-testid="stAlert"] {{
        background: var(--alert-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--alert-text) !important;
    }}

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {{
        color: var(--alert-text) !important;
    }}

    /* BUTTONS */

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {{
        border-radius: 12px;
        border: 1px solid var(--border);
        background: linear-gradient(
            110deg,
            var(--accent),
            var(--accent2)
        );
        color: white !important;
        font-weight: 700;
        min-height: 42px;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {{
        filter: brightness(1.12);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px var(--shadow);
    }}

    /* TABS */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: var(--surface);
        padding: 7px;
        border: 1px solid var(--border);
        border-radius: 14px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        color: var(--muted) !important;
        font-weight: 600;
        padding: 10px 18px;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--surface-hover) !important;
        color: var(--accent) !important;
    }}

    /* CHAT */

    [data-testid="stChatMessage"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 15px;
        margin-bottom: 12px;
    }}

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] [data-baseweb="textarea"] {{
        background: var(--input-bg) !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 12px !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
    }}

    /* CONTAINERS */

    [data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--surface);
        border-color: var(--border) !important;
        border-radius: 13px;
    }}

    /* PROGRESS */

    [data-testid="stProgressBar"] > div {{
        background: var(--progress-track) !important;
        border-radius: 20px !important;
    }}

    [data-testid="stProgressBar"] [role="progressbar"] {{
        background: var(--progress-track) !important;
    }}

    [data-testid="stProgressBar"] [role="progressbar"] > div {{
        background: linear-gradient(
            90deg,
            var(--accent),
            var(--accent2)
        ) !important;
        border-radius: 20px !important;
    }}

    hr {{
        border-color: var(--border);
    }}

    .footer {{
        text-align: center;
        color: var(--muted);
        font-size: 12px;
        padding: 25px 0 5px;
    }}


    /* ==================================================
       QUILLQUERY — LUXURY / MONOCHROME DESIGN SYSTEM
       ================================================== */

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif !important;
    }}

    h1, h2, h3, h4, .hero-title, .brand {{
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em !important;
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background:
            radial-gradient(ellipse at 78% 0%, rgba(220,220,220,0.055), transparent 34%),
            #090A0C !important;
    }}

    .block-container {{
        max-width: 1380px;
        padding-top: 2.6rem;
        padding-bottom: 4rem;
    }}

    [data-testid="stSidebar"] {{
        background: #101114 !important;
        border-right: 1px solid rgba(220,220,220,0.11) !important;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 23px;
        font-weight: 800;
        color: #F2F2F0;
        letter-spacing: -1.1px !important;
    }}

    .brand-mark {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 29px;
        height: 29px;
        border: 1px solid rgba(220,220,220,0.42);
        border-radius: 9px;
        background: rgba(220,220,220,0.10);
        color: #D8D8D8 !important;
        font-size: 15px;
        letter-spacing: -0.04em;
    }}

    .brand span:last-child {{
        color: #D8D8D8 !important;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 38px 40px;
        border-radius: 18px;
        border: 1px solid rgba(220,220,220,0.16);
        background:
            linear-gradient(112deg, rgba(48,50,54,0.62), rgba(19,20,23,0.94) 62%),
            #131416 !important;
        box-shadow: 0 24px 70px rgba(0,0,0,0.22);
        backdrop-filter: blur(16px);
        margin-bottom: 26px;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: -110px;
        top: -180px;
        border: 1px solid rgba(220,220,220,0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 32px rgba(220,220,220,0.025),
                    0 0 0 66px rgba(220,220,220,0.018);
        pointer-events: none;
    }}

    .hero-label {{
        color: #D8D8D8 !important;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2.4px;
    }}

    .hero-title {{
        max-width: 700px;
        font-size: clamp(34px, 4vw, 48px);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -2.1px !important;
        color: #F2F2F0 !important;
        margin-top: 18px;
    }}

    .hero-description {{
        max-width: 650px;
        color: #B5B5B5 !important;
        font-size: 14px;
        line-height: 1.8;
        margin-top: 15px;
    }}

    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018)) !important;
        border: 1px solid rgba(220,220,220,0.12) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.10);
    }}

    [data-testid="stMetricLabel"] {{
        color: #A1A1A1 !important;
        font-size: 11px !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    [data-testid="stMetricValue"] {{
        color: #F2F2F0 !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 29px !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 5px !important;
        padding: 6px !important;
        border: 1px solid rgba(220,220,220,0.12) !important;
        border-radius: 13px !important;
        background: rgba(255,255,255,0.025) !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px !important;
        padding: 10px 17px !important;
        color: #A1A1A1 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: rgba(220,220,220,0.10) !important;
        color: #F0F0F0 !important;
    }}

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {{
        background: #77797D !important;
        border: 1px solid rgba(230,230,230,0.22) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        box-shadow: none !important;
        transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {{
        background: #929498 !important;
        border-color: rgba(240,240,240,0.45) !important;
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(220,220,220,0.10) !important;
    }}

    .stTextInput input,
    .stTextArea textarea,
    [data-testid="stChatInput"] textarea {{
        background: #131416 !important;
        border: 1px solid rgba(220,220,220,0.16) !important;
        border-radius: 10px !important;
    }}

    [data-testid="stFileUploader"] {{
        border: 1px dashed rgba(105,201,155,0.40) !important;
        border-radius: 12px !important;
        background: rgba(220,220,220,0.025) !important;
    }}

    [data-testid="stChatMessage"] {{
        background: rgba(255,255,255,0.028) !important;
        border: 1px solid rgba(220,220,220,0.11) !important;
        border-radius: 13px !important;
    }}

    [data-testid="stExpander"],
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: rgba(220,220,220,0.13) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.022) !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 11px !important;
    }}

    .footer {{
        border-top: 1px solid rgba(220,220,220,0.10);
        margin-top: 34px;
        padding-top: 22px;
        color: #85878B !important;
        letter-spacing: 0.08em;
        font-size: 10px !important;
    }}

    /* Keep native Streamlit controls readable in the dark-only workspace. */
    [data-baseweb="select"] *,
    [role="radiogroup"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {{
        color: #E6E6E6 !important;
    }}

    [data-testid="stProgressBar"] [role="progressbar"] > div {{
        background: linear-gradient(90deg, #77797D, #E8E8E8) !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 5. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="brand"><span class="brand-mark">Q</span> Quill<span>Query</span></div>',
        unsafe_allow_html=True
    )

    st.caption("Every document has an answer.")

    st.divider()

    st.markdown("### 📁 Document Library")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload digital PDFs or TXT files."
    )

    if st.button("⚡ Process Documents", width="stretch"):

        if not uploaded_files:

            st.warning("Upload at least one document first.")

        else:

            with st.spinner("Processing your documents..."):

                temp_folder = Path("temp_documents")
                temp_folder.mkdir(exist_ok=True)

                documents = []

                try:

                    for file in uploaded_files:

                        file_path = temp_folder / file.name
                        file_path.write_bytes(file.getvalue())

                        documents.extend(load_document(file_path))

                    if not documents:

                        st.error("No readable text found.")

                    else:

                        (
                            vector_store,
                            bm25_retriever,
                            total_chunks
                        ) = create_vector_store(documents)

                        st.session_state.vector_store = vector_store
                        st.session_state.bm25_retriever = bm25_retriever
                        st.session_state.messages = []

                        st.session_state.uploaded_files = [
                            file.name for file in uploaded_files
                        ]

                        st.session_state.quiz_questions = []
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}

                        st.success(
                            f"Successfully processed "
                            f"{len(documents)} pages into "
                            f"{total_chunks} chunks!"
                        )

                except Exception as e:

                    st.error(f"Processing failed: {e}")

    st.divider()

    st.markdown("### Processed Documents")

    if st.session_state.uploaded_files:

        for filename in st.session_state.uploaded_files:
            st.markdown(f"📄 `{filename}`")

    else:

        st.caption("No documents processed yet.")

    st.divider()

    st.markdown(
        """
        <div class="sidebar-note">
        <b>QUILLQUERY</b><br>
        AI-powered document intelligence.<br><br>
        Powered by Hybrid RAG, FAISS,
        BM25 and Groq.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 6. HERO
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">
            DOCUMENT INTELLIGENCE · YOUR KNOWLEDGE WORKSPACE
        </div>
        <div class="hero-title">
            Knowledge begins<br>with a better question.
        </div>
        <div class="hero-description">
            Ask questions, generate quizzes, and understand
            your learning progress — all in one workspace.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 7. QUICK STATUS
# ==================================================

stats = get_performance()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents loaded", len(st.session_state.uploaded_files))

with col2:
    st.metric("Quiz attempts", stats["total_attempts"])

with col3:
    st.metric("Average score", f"{stats['average_score']:.1f}%")

st.divider()


# ==================================================
# 8. WORKSPACE TABS
# ==================================================

tab_chat, tab_quiz, tab_analytics = st.tabs([
    "💬  Ask Documents",
    "📝  Quiz Studio",
    "📊  Analytics"
])


# ==================================================
# 9. DOCUMENT CHAT
# ==================================================

with tab_chat:

    st.markdown("### 💬 Ask Your Documents Anything")

    st.caption(
        "Ask questions in natural language. "
        "QuillQuery retrieves relevant information from your documents."
    )

    if st.session_state.vector_store is None:

        st.info(
            "📁 Upload and process your documents using the sidebar "
            "to start asking questions."
        )

    else:

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message.get("sources"):

                    with st.expander("📚 View Sources"):

                        for source, page in message["sources"]:
                            st.write(f"📄 **{source}** — Page {page}")

        query = st.chat_input("Ask something about your documents...")

        if query:

            st.session_state.messages.append({
                "role": "user",
                "content": query
            })

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):

                with st.spinner("Searching your documents..."):

                    try:

                        answer, sources = answer_question(
                            st.session_state.vector_store,
                            st.session_state.bm25_retriever,
                            query
                        )

                        st.markdown(answer)

                        if sources:

                            with st.expander("📚 View Sources"):

                                for source, page in sources:
                                    st.write(
                                        f"📄 **{source}** — Page {page}"
                                    )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    except Exception as e:

                        st.error(f"Something went wrong: {e}")


# ==================================================
# 10. QUIZ STUDIO
# ==================================================

with tab_quiz:

    st.markdown("### 📝 Quiz Studio")

    st.caption(
        "Turn your documents into interactive knowledge tests."
    )

    if st.session_state.vector_store is None:

        st.info("Upload and process documents before generating quizzes.")

    else:

        with st.container(border=True):

            st.markdown("#### Create a new quiz")

            with st.form("quiz_generation_form"):

                topic = st.text_input(
                    "What topic do you want to practise?",
                    placeholder="e.g. Design and Analysis of Algorithms"
                )

                number_of_questions = st.selectbox(
                    "Number of questions",
                    [5, 10]
                )

                generate_button = st.form_submit_button(
                    "✨ Generate Quiz",
                    width="stretch"
                )

            if generate_button:

                if not topic.strip():

                    st.warning("Please enter a topic.")

                else:

                    with st.spinner("Generating your questions..."):

                        try:

                            questions = generate_mcqs(
                                st.session_state.vector_store,
                                topic,
                                number_of_questions
                            )

                            if not questions:

                                st.warning(
                                    "Not enough relevant information. "
                                    "Try another topic."
                                )

                            else:

                                st.session_state.quiz_questions = questions
                                st.session_state.quiz_topic = topic
                                st.session_state.quiz_submitted = False
                                st.session_state.quiz_answers = {}
                                st.session_state.quiz_score = 0

                                st.rerun()

                        except Exception as e:

                            st.error(f"Quiz generation failed: {e}")

        if st.session_state.quiz_questions:

            questions = st.session_state.quiz_questions

            st.divider()

            st.markdown(f"### Quiz: {st.session_state.quiz_topic}")
            st.caption(f"{len(questions)} questions")

            if not st.session_state.quiz_submitted:

                with st.form("quiz_answer_form"):

                    selected_answers = {}

                    for i, question in enumerate(questions):

                        st.markdown(f"**Question {i + 1}**")
                        st.markdown(question["question"])

                        selected_answers[i] = st.radio(
                            "Choose your answer:",
                            options=["A", "B", "C", "D"],
                            format_func=lambda x, q=question:
                                f"{x}. {q['options'][x]}",
                            key=f"quiz_answer_{i}"
                        )

                        st.divider()

                    submit_quiz = st.form_submit_button(
                        "Submit Quiz",
                        width="stretch"
                    )

                if submit_quiz:

                    score = sum(
                        selected_answers[i] == question["answer"]
                        for i, question in enumerate(questions)
                    )

                    st.session_state.quiz_answers = selected_answers
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True

                    save_quiz_result(
                        st.session_state.quiz_topic,
                        score,
                        len(questions)
                    )

                    st.rerun()

            else:

                score = st.session_state.quiz_score
                total = len(questions)
                percentage = (score / total) * 100

                st.markdown("### 📊 Quiz Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Your Score", f"{score}/{total}")

                with col2:
                    st.metric("Accuracy", f"{percentage:.1f}%")

                st.progress(percentage / 100)

                if percentage >= 80:
                    st.success("Excellent work! 🎉")
                elif percentage >= 50:
                    st.info("Good effort! Keep practising.")
                else:
                    st.warning("Keep learning. You'll improve!")

                st.divider()
                st.markdown("### Answer Review")

                for i, question in enumerate(questions):

                    selected = st.session_state.quiz_answers[i]
                    correct = question["answer"]

                    with st.container(border=True):

                        st.markdown(
                            f"**Q{i + 1}. {question['question']}**"
                        )

                        if selected == correct:

                            st.success(
                                f"Correct! Your answer: {selected}"
                            )

                        else:

                            st.error(
                                f"Incorrect. Your answer: {selected}"
                            )

                            st.info(
                                f"Correct answer: {correct}. "
                                f"{question['options'][correct]}"
                            )

                        st.caption(
                            f"Explanation: {question['explanation']}"
                        )

                if st.button("↻ Generate New Quiz", width="stretch"):

                    st.session_state.quiz_questions = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_score = 0

                    st.rerun()


# ==================================================
# 11. LEARNING ANALYTICS
# ==================================================

with tab_analytics:

    st.markdown("### 📊 Learning Analytics")

    st.caption(
        "Understand your progress and identify areas that need practice."
    )

    stats = get_performance()
    history = get_quiz_history()
    weak_topics = get_weak_topics()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Attempts", stats["total_attempts"])

    with col2:
        st.metric("Average Score", f"{stats['average_score']:.1f}%")

    with col3:
        st.metric("Topics to Improve", len(weak_topics))

    st.divider()

    if history:

        df = pd.DataFrame(history)

        df["attempted_at"] = pd.to_datetime(
            df["attempted_at"],
            errors="coerce"
        )

        df = df.dropna(subset=["attempted_at"])
        df = df.sort_values("attempted_at")

        axis_theme = alt.Axis(
            labelColor=chart_text,
            titleColor=chart_text,
            gridColor=chart_grid,
            domainColor=chart_grid,
            tickColor=chart_grid,
            labelFont="DM Sans",
            titleFont="DM Sans"
        )

        # PERFORMANCE OVER TIME

        st.markdown("#### 📈 Performance Over Time")

        trend_chart = (
            alt.Chart(df)
            .mark_line(
                color=accent2,
                strokeWidth=3,
                point=alt.OverlayMarkDef(
                    filled=True,
                    color=accent
                )
            )
            .encode(
                x=alt.X(
                    "attempted_at:T",
                    title="Attempt date",
                    axis=axis_theme
                ),
                y=alt.Y(
                    "percentage:Q",
                    title="Score (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=axis_theme
                ),
                tooltip=[
                    alt.Tooltip("topic:N", title="Topic"),
                    alt.Tooltip(
                        "percentage:Q",
                        title="Score",
                        format=".1f"
                    ),
                    alt.Tooltip(
                        "attempted_at:T",
                        title="Date"
                    )
                ]
            )
            .properties(
                height=320,
                background=chart_bg
            )
            .configure_view(strokeOpacity=0)
            .configure_axis(
                grid=True,
                labelFontSize=12,
                titleFontSize=13
            )
        )

        st.altair_chart(
            trend_chart,
            width="stretch",
            theme=None
        )

        st.divider()

        # TOPIC-WISE PERFORMANCE

        st.markdown("#### 📚 Topic-wise Performance")

        topic_df = (
            df.groupby("topic", as_index=False)["percentage"]
            .mean()
            .sort_values("percentage", ascending=False)
        )

        topic_chart = (
            alt.Chart(topic_df)
            .mark_bar(
                color=accent2,
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            )
            .encode(
                x=alt.X(
                    "topic:N",
                    title="Topic",
                    sort="-y",
                    axis=alt.Axis(
                        labelColor=chart_text,
                        titleColor=chart_text,
                        labelAngle=-30,
                        labelFont="DM Sans",
                        titleFont="DM Sans"
                    )
                ),
                y=alt.Y(
                    "percentage:Q",
                    title="Average score (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=axis_theme
                ),
                tooltip=[
                    alt.Tooltip("topic:N", title="Topic"),
                    alt.Tooltip(
                        "percentage:Q",
                        title="Average score",
                        format=".1f"
                    )
                ]
            )
            .properties(
                height=320,
                background=chart_bg
            )
            .configure_view(strokeOpacity=0)
            .configure_axis(
                grid=True,
                labelFontSize=12,
                titleFontSize=13
            )
        )

        st.altair_chart(
            topic_chart,
            width="stretch",
            theme=None
        )

        st.divider()

        # TOPICS TO IMPROVE

        st.markdown("#### 🎯 Topics to Improve")

        if weak_topics:

            for topic, average_score, attempts in weak_topics:

                with st.container(border=True):

                    st.markdown(f"**{topic}**")

                    st.write(
                        f"Average score: {average_score:.1f}% "
                        f"• {attempts} attempts"
                    )

                    st.progress(
                        min(max(average_score / 100, 0), 1)
                    )

        else:

            st.success(
                "No weak topics identified yet. "
                "Keep attempting quizzes to build your learning history."
            )

        st.divider()

        # QUIZ HISTORY TABLE

        st.markdown("#### 📋 Quiz History")

        display_df = df[[
            "topic",
            "score",
            "total_questions",
            "percentage",
            "attempted_at"
        ]].copy()

        display_df.columns = [
            "Topic",
            "Score",
            "Total Questions",
            "Percentage",
            "Date"
        ]

        display_df["Date"] = (
            display_df["Date"].dt.strftime("%d %b %Y, %I:%M %p")
        )

        # Theme-aware table styling
        styled_df = (
            display_df.style
            .set_properties(
                **{
                    "background-color": table_row,
                    "color": text,
                    "border-color": border,
                    "padding": "10px"
                }
            )
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", table_header),
                        ("color", text),
                        ("font-weight", "700"),
                        ("border-color", border),
                        ("padding", "10px")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-color", border),
                        ("padding", "10px")
                    ]
                }
            ])
        )

        # No unsupported theme argument
        st.dataframe(
            styled_df,
            width="stretch",
            hide_index=True
        )

        csv = display_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Quiz History",
            data=csv,
            file_name="quillquery_quiz_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Your analytics will appear here after your first quiz attempt."
        )


# ==================================================
# 12. FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        QUILLQUERY · EVERY DOCUMENT HAS AN ANSWER.
        <br>
        A considered workspace for the documents that matter.
    </div>
    """,
    unsafe_allow_html=True
)
