import json
import os
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import streamlit as st

# Step 4.1: Connect to Llama 3.1:8B via Ollama
llm = Ollama(
    model="llama3.1:8b",  # Specify the model name as per Ollama
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7  # Control creativity
)

# Step 4.2: Define Long-Term Memory Handling
MEMORY_FILE = "long_term_memory.json"

def load_long_term_memory(user_id):
    """Load user-specific long-term memory from JSON file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            return data.get(user_id, {"preferences": {}, "history": []})
    return {"preferences": {}, "history": []}

def save_long_term_memory(user_id, memory_data):
    """Save user-specific long-term memory to JSON file."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[user_id] = memory_data
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Step 4.3: Custom Prompt Template for Hotel Booking with Long-Term Context
template = """You are a hotel booking assistant. Help the user book a hotel by asking relevant questions (e.g., location, dates, budget). Remember their preferences and past interactions if available.

Long-Term Context (User Preferences and Past Bookings):
{long_term_context}

Current Conversation History:
{history}

User Input: {input}

Assistant: """
PROMPT = PromptTemplate(input_variables=["long_term_context", "history", "input"], template=template)

# Step 4.4: Streamlit Interface and Chatbot Logic
def main():
    st.title("Hotel Booking Chatbot with Long-Term Memory")
    st.write("This chatbot remembers your preferences and past bookings across sessions!")

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
    long_term_context_str = f"Preferences: {long_term_data['preferences']}\nPast Bookings: {long_term_data['history']}"

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