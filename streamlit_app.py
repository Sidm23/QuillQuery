import pandas as pd
import streamlit as st
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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="QuillQuery",
    page_icon="📚",
    layout="wide"
)


# Initialize database
initialize_database()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

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


# --------------------------------------------------
# SIDEBAR - DOCUMENT UPLOAD
# --------------------------------------------------

with st.sidebar:

    st.title("📚 QuillQuery")
    st.caption("Every document has an answer.")

    st.divider()

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if not uploaded_files:

            st.warning("Please upload at least one document.")

        else:

            with st.spinner("Reading and processing documents..."):

                temp_folder = Path("temp_documents")
                temp_folder.mkdir(exist_ok=True)

                documents = []

                try:

                    for file in uploaded_files:

                        file_path = temp_folder / file.name
                        file_path.write_bytes(file.getvalue())

                        documents.extend(load_document(file_path))

                    if not documents:

                        st.error("No readable text found in uploaded files.")

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
                            f"Processed {len(documents)} pages/documents "
                            f"into {total_chunks} chunks!"
                        )

                except Exception as e:

                    st.error(f"Processing failed: {e}")


    # Processed documents
    if st.session_state.uploaded_files:

        st.divider()
        st.subheader("Processed Documents")

        for filename in st.session_state.uploaded_files:
            st.write(f"📄 {filename}")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("QuillQuery")
st.caption("Your AI-powered document intelligence assistant.")

st.divider()


if st.session_state.vector_store is None:

    st.info(
        "👈 Upload your PDF or TXT documents from the sidebar "
        "and click Process Documents to get started."
    )

else:

    # --------------------------------------------------
    # DOCUMENT CHAT
    # --------------------------------------------------

    st.subheader("💬 Ask Your Documents Anything")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("View Sources"):

                    for source, page in message["sources"]:
                        st.write(f"📄 {source} — Page {page}")


    query = st.chat_input(
        "Ask a question about your documents..."
    )

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

                        with st.expander("View Sources"):

                            for source, page in sources:
                                st.write(f"📄 {source} — Page {page}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(f"Something went wrong: {e}")


    # --------------------------------------------------
    # MCQ QUIZ
    # --------------------------------------------------

    st.divider()

    st.header("📝 Test Your Knowledge")

    st.write(
        "Generate a quiz from your uploaded documents "
        "and test what you've learned."
    )

    with st.form("quiz_generation_form"):

        topic = st.text_input(
            "Enter a topic",
            placeholder="e.g. Design and Analysis of Algorithms"
        )

        number_of_questions = st.selectbox(
            "Number of questions",
            [5, 10]
        )

        generate_button = st.form_submit_button(
            "Generate Quiz",
            use_container_width=True
        )

    if generate_button:

        if not topic.strip():

            st.warning("Please enter a topic.")

        else:

            with st.spinner("Generating your quiz..."):

                try:

                    questions = generate_mcqs(
                        st.session_state.vector_store,
                        topic,
                        number_of_questions
                    )

                    if not questions:

                        st.warning(
                            "Not enough relevant information found. "
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


    # --------------------------------------------------
    # DISPLAY QUIZ
    # --------------------------------------------------

    if st.session_state.quiz_questions:

        questions = st.session_state.quiz_questions

        st.divider()

        st.subheader(
            f"Quiz: {st.session_state.quiz_topic}"
        )

        st.caption(f"{len(questions)} questions")

        if not st.session_state.quiz_submitted:

            with st.form("quiz_answer_form"):

                selected_answers = {}

                for i, question in enumerate(questions):

                    st.markdown(
                        f"**Q{i + 1}. {question['question']}**"
                    )

                    selected_answers[i] = st.radio(
                        "Select your answer:",
                        options=["A", "B", "C", "D"],
                        format_func=lambda x, q=question:
                            f"{x}. {q['options'][x]}",
                        key=f"quiz_answer_{i}"
                    )

                    st.divider()

                submit_quiz = st.form_submit_button(
                    "Submit Quiz",
                    use_container_width=True
                )

            if submit_quiz:

                score = sum(
                    selected_answers[i] == question["answer"]
                    for i, question in enumerate(questions)
                )

                st.session_state.quiz_answers = selected_answers
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True

                # Save result permanently
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

            st.subheader("📊 Your Results")

            st.metric(
                "Your Score",
                f"{score}/{total}",
                f"{percentage:.1f}%"
            )

            if percentage >= 80:
                st.success("Excellent work! 🎉")

            elif percentage >= 50:
                st.info("Good effort! Keep practising.")

            else:
                st.warning("Keep learning. You'll improve!")

            st.divider()

            st.subheader("Answer Review")

            for i, question in enumerate(questions):

                selected = st.session_state.quiz_answers[i]
                correct = question["answer"]

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

                st.write(
                    f"**Explanation:** {question['explanation']}"
                )

                st.divider()

            if st.button(
                "Generate New Quiz",
                use_container_width=True
            ):

                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.quiz_score = 0

                st.rerun()


# --------------------------------------------------
# PERFORMANCE DASHBOARD
# --------------------------------------------------

st.divider()

st.header("📈 Learning Performance Dashboard")

stats = get_performance()
history = get_quiz_history()
weak_topics = get_weak_topics()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Quiz Attempts",
        stats["total_attempts"]
    )

with col2:
    st.metric(
        "Average Score",
        f"{stats['average_score']:.1f}%"
    )

with col3:
    st.metric(
        "Topics to Improve",
        len(weak_topics)
    )


# Weak topics
st.subheader("🎯 Topics to Improve")

if weak_topics:

    for topic, average_score, attempts in weak_topics:

        st.write(
            f"**{topic}** — Average: {average_score:.1f}% "
            f"({attempts} attempts)"
        )

        st.progress(
            min(max(average_score / 100, 0.0), 1.0)
        )

else:

    st.success(
        "No weak topics identified yet. "
        "Complete more quizzes to track your progress."
    )



# Quiz history and analytics
st.subheader("📚 Quiz History")

if history:

    df = pd.DataFrame(history)

    # Score trend
    st.markdown("#### Performance Over Time")

    trend_df = df.sort_values("attempted_at")

    st.line_chart(
        trend_df.set_index("attempted_at")["percentage"]
    )

    # Topic-wise performance
    st.markdown("#### Topic-wise Performance")

    topic_df = (
        df.groupby("topic", as_index=False)["percentage"]
        .mean()
        .sort_values("percentage", ascending=False)
    )

    st.bar_chart(
        topic_df.set_index("topic")["percentage"]
    )

    # Detailed history
    st.markdown("#### All Attempts")

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

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Download history
    csv = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Quiz History",
        data=csv,
        file_name="quillquery_quiz_history.csv",
        mime="text/csv"
    )

else:

    st.info("Your quiz attempts will appear here.")
