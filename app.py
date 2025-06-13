import streamlit as st
import os
from time import sleep
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

# SQLite setup
Base = declarative_base()
class UserProfile(Base):
    __tablename__ = 'user_profiles'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)
    preferences = Column(Text)

engine = create_engine('sqlite:///chatbot.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# LLM retry logic
def call_llm_with_retry(runnable, messages, config, max_retries=3):
    for attempt in range(max_retries):
        try:
            return runnable.invoke(messages, config=config)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                sleep(2 ** attempt)
            else:
                raise e
    raise Exception("Max retries exceeded")

# Chat history store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Fallback parsing for room type
def parse_room_type(history_str):
    history_lower = history_str.lower()
    room_types = ["king bed", "double bed", "queen bed", "suite", "single bed"]
    for room in room_types:
        if room in history_lower:
            return room.capitalize()
    return "Not Specified"

# Streamlit UI
st.title("Hotel Booking Chatbot")
st.write("Enter your User ID and ask about hotels. I’ll remember your preferences across sessions!")

# User ID and name input
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
user_id = st.text_input("Enter your User ID (e.g., user_1):", value=st.session_state.user_id)
if user_id:
    st.session_state.user_id = user_id

if "user_name" not in st.session_state:
    st.session_state.user_name = ""
user_name = st.text_input("Enter your name (e.g., John):", value=st.session_state.user_name)
if user_name:
    st.session_state.user_name = user_name

# Initialize LLM and chain
if user_id and "chain" not in st.session_state:
    # Load preferences
    session = Session()
    user = session.query(UserProfile).filter_by(user_id=user_id).first()
    preferences = user.preferences if user else f"Name: {user_name}" if user_name else "No preferences found."
    st.session_state.preferences = preferences
    session.close()

    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a hotel booking assistant. Use the following user preferences if relevant: {preferences}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini") if USE_OPENAI else ChatOllama(model="qwen2.5", temperature=0.7)

    # Create chain
    chain = prompt | llm
    st.session_state.chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# Display conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_id and (user_input := st.chat_input("What would you like to know about hotels?")):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        config = {"configurable": {"session_id": user_id}}
        messages = {"input": user_input, "preferences": st.session_state.preferences}
        response = call_llm_with_retry(st.session_state.chain, messages, config)
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Summarize and save preferences
    session = Session()
    history = get_session_history(user_id).messages
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    
    # Debug: Log raw history
    st.write("Debug - Raw Conversation History:", history_str)

    summary_prompt = f"""
    Summarize the conversation to extract user preferences in this exact format: 
    "Name: <name>, Location: <location>, Room Type: <room type>, Other: <other preferences>"
    - Name: Use the provided user name or 'Not Provided' if none.
    - Location: Identify the city or area (e.g., Paris, New York).
    - Room Type: Identify the room type (e.g., King Bed, Double Bed, Suite). Do not use 'Not Specified' unless no room type is mentioned.
    - Other: Include other preferences (e.g., Near Eiffel Tower, Has Pool).
    Examples:
    - "I want a hotel in Paris with a king bed near the Eiffel Tower" → "Name: Not Provided, Location: Paris, Room Type: King Bed, Other: Near Eiffel Tower"
    - "I'm John, I need a suite in London with a pool" → "Name: John, Location: London, Room Type: Suite, Other: Has Pool"
    Conversation:
    {history_str}
    User name: {user_name}
    """
    summary_llm = ChatOllama(model="qwen2.5")
    summary_response = summary_llm.invoke(summary_prompt)
    summary = summary_response.content

    # Fallback: Parse room type if summary fails
    if "Room Type: Not Specified" in summary:
        room_type = parse_room_type(history_str)
        summary = summary.replace("Room Type: Not Specified", f"Room Type: {room_type}")

    # Debug: Log summary
    st.write("Debug - Summary Generated:", summary)

    user = session.query(UserProfile).filter_by(user_id=user_id).first()
    if user:
        user.preferences = summary
    else:
        user = UserProfile(user_id=user_id, preferences=summary)
        session.add(user)
    session.commit()
    st.session_state.preferences = summary
    session.close()

# Display memory
if user_id:
    history = get_session_history(user_id).messages
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    with st.expander("View Memory"):
        st.write("Short-Term Memory:", history_str)
        st.write("Long-Term Memory (Preferences):", st.session_state.preferences)