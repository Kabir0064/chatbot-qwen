import sqlite3
import json
import os
from datetime import datetime
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import streamlit as st

# Step 1: Connect to Llama 3.1:8B via Ollama
llm = Ollama(
    model="llama3.1:8b",  # Specify the model name as per Ollama
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7  # Control creativity
)

# Step 2: SQLite Database Setup for Long-Term Memory
DB_FILE = "long_term_memory.db"

def init_database():
    """Initialize SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id TEXT PRIMARY KEY,
            created_at TEXT
        )
    ''')
    
    # Create Memory table for storing preferences and history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            data_type TEXT,
            key TEXT,
            value TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES Users(user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_long_term_memory(user_id):
    """Load user-specific long-term memory from SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Ensure user exists in Users table
    cursor.execute('INSERT OR IGNORE INTO Users (user_id, created_at) VALUES (?, ?)',
                   (user_id, datetime.now().isoformat()))
    conn.commit()
    
    # Fetch memory data for the user
    cursor.execute('SELECT data_type, key, value FROM Memory WHERE user_id = ?', (user_id,))
    rows = cursor.fetchall()
    
    # Structure data into preferences and history
    preferences = {}
    history = []
    for row in rows:
        data_type, key, value = row
        if data_type == "preference":
            preferences[key] = value
        elif data_type == "history":
            try:
                history.append(json.loads(value))  # Deserialize history entry
            except json.JSONDecodeError:
                continue  # Skip if invalid JSON
    
    conn.close()
    return {"preferences": preferences, "history": history}

def save_long_term_memory(user_id, memory_data):
    """Save user-specific long-term memory to SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Ensure user exists in Users table
    cursor.execute('INSERT OR IGNORE INTO Users (user_id, created_at) VALUES (?, ?)',
                   (user_id, datetime.now().isoformat()))
    
    # Save preferences
    for key, value in memory_data.get("preferences", {}).items():
        cursor.execute('''
            INSERT OR REPLACE INTO Memory (user_id, data_type, key, value, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, "preference", key, value, datetime.now().isoformat()))
    
    # Save history (serialize each entry as JSON)
    for entry in memory_data.get("history", []):
        # Check if this entry already exists to avoid duplicates
        serialized_entry = json.dumps(entry)
        cursor.execute('''
            SELECT id FROM Memory
            WHERE user_id = ? AND data_type = ? AND value = ?
        ''', (user_id, "history", serialized_entry))
        if not cursor.fetchone():  # Only insert if not already present
            cursor.execute('''
                INSERT INTO Memory (user_id, data_type, key, value, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, "history", f"interaction_{len(memory_data['history'])}", serialized_entry, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

# Step 3: Custom Prompt Template for Hotel Booking with Enhanced Context Utilization
template = """You are a hotel booking assistant. Your goal is to help the user book a hotel by asking relevant questions (e.g., location, dates, budget). Always reference and utilize the user's stored preferences and past interactions from the long-term context when relevant to the current query. If the user has previously mentioned preferences (like budget or location), acknowledge them in your response or confirm if they still apply before proceeding.

Long-Term Context (User Preferences and Past Interactions):
{long_term_context}

Current Conversation History:
{history}

User Input: {input}

Assistant: """
PROMPT = PromptTemplate(input_variables=["long_term_context", "history", "input"], template=template)

# Step 4: Streamlit Interface and Chatbot Logic
def main():
    st.title("Hotel Booking Chatbot with Long-Term Memory")
    st.write("This chatbot remembers your preferences and past bookings across sessions!")

    # Initialize database
    init_database()
    
    # Initialize session state for user ID and conversation
    if "user_id" not in st.session_state:
        st.session_state.user_id = st.text_input("Enter your User ID (e.g., user123):", value="user123")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(ai_prefix="Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_id = st.session_state.user_id

    # Load long-term memory for this user
    long_term_data = load_long_term_memory(user_id)
    
    # Format long-term context in a concise, structured way for the prompt
    preferences = long_term_data.get("preferences", {})
    history = long_term_data.get("history", [])
    
    # Structure preferences for clarity in prompt
    pref_str = "None available"
    if preferences:
        pref_str = "\n- " + "\n- ".join([f"{key}: {value}" for key, value in preferences.items()])
    
    # Limit history to the last 3 interactions to avoid overwhelming the prompt
    hist_str = "None available"
    if history:
        recent_history = history[-3:]  # Take last 3 interactions if available
        hist_str = "\n- " + "\n- ".join([f"User: {item['user_input']} | Assistant: {item['assistant_response']}" for item in recent_history])
    
    long_term_context_str = f"Stored Preferences:\n{pref_str}\n\nRecent Past Interactions:\n{hist_str}"

    # Initialize conversation chain with long-term context
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=PROMPT.partial(long_term_context=long_term_context_str)
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask me about booking a hotel..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = conversation.predict(input=prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Update long-term memory (example: store user input as history)
        long_term_data["history"].append({"user_input": prompt, "assistant_response": response})
        if "budget" in prompt.lower():
            long_term_data["preferences"]["budget"] = prompt  # Basic parsing example
        if "location" in prompt.lower() or "city" in prompt.lower():
            long_term_data["preferences"]["location"] = prompt  # Basic parsing example
        save_long_term_memory(user_id, long_term_data)

if __name__ == "__main__":
    main()