
import sqlite3
from datetime import datetime


DB_NAME = "quillquery.db"


# Create database and table
def initialize_database():

    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            percentage REAL NOT NULL,
            attempted_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# Save a quiz attempt
def save_quiz_result(topic, score, total_questions):

    percentage = (score / total_questions) * 100

    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO quiz_history
        (topic, score, total_questions, percentage, attempted_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        topic,
        score,
        total_questions,
        percentage,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# Fetch all quiz attempts
def get_quiz_history():

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM quiz_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# Get performance statistics
def get_performance():

    conn = sqlite3.connect(DB_NAME)

    stats = conn.execute("""
        SELECT
            COUNT(*) AS total_attempts,
            AVG(percentage) AS average_score
        FROM quiz_history
    """).fetchone()

    conn.close()

    return {
        "total_attempts": stats[0],
        "average_score": stats[1] or 0
    }


# Identify weak topics
def get_weak_topics():

    conn = sqlite3.connect(DB_NAME)

    rows = conn.execute("""
        SELECT
            topic,
            AVG(percentage) AS average_score,
            COUNT(*) AS attempts
        FROM quiz_history
        GROUP BY LOWER(topic)
        HAVING AVG(percentage) < 60
        ORDER BY average_score ASC
    """).fetchall()

    conn.close()

    return rows
