# Project Journey: Context Management in LLM-based ChatBot for Hotel Booking

## Project Overview

**Assignment Title:** Context Management in LLM-based ChatBots using LangChain
**Problem Definition:** LLMs like Llama 3.1:8B are stateless and require explicit context management to remember previous user interactions. This project aims to implement long-term memory to maintain coherence across multi-turn conversations in a hotel booking assistant.
**Objectives and Deliverables:**

* Implement long-term context memory
* Use LangChain with Llama 3.1:8B hosted via Ollama
* Create a Streamlit interface
* Demonstrate memory-aware hotel booking interactions

---

## Development Journey

### 1. Requirement Analysis

* Identified need for persistent memory in LLM conversations
* Chose hotel booking assistant for natural multi-turn interaction
* Selected LangChain + Ollama (Llama 3.1:8B) + Streamlit tech stack

### 2. Initial Long-Term Memory (JSON)

* Stored user preferences and chat history in `long_term_memory.json`
* Built `load_long_term_memory()` and `save_long_term_memory()`
* Added error handling for JSON decoding
* Integrated with LangChain’s `ConversationBufferMemory`

### 3. Enhanced Prompt Engineering

* Structured prompt with stored preferences + recent history
* Prompted LLM to acknowledge and confirm past data
* Example: "I remember you wanted a hotel in Paris. Is that correct?"

### 4. Migrated to SQLite

* Replaced JSON with `long_term_memory.db` for better scalability
* Tables:

  * `Users`: stores user\_id
  * `Memory`: stores data\_type (preferences/history), key, value
* Structured memory format with serialized history
* Used `INSERT OR REPLACE` for managing updates

---

## Architecture

```
User (Streamlit UI)
      ↓
LangChain (ConversationChain with PromptTemplate)
      ↓
Long-Term Memory (SQLite)
      ↓
LLM Backend (Llama 3.1:8B via Ollama)
```

---

## Workflow Summary

1. User logs in with a user\_id
2. App loads long-term context from SQLite
3. LangChain builds prompt with session + long-term context
4. Llama 3.1:8B generates memory-aware response
5. New context is saved back into SQLite

---

## Demo Scenario

User: "I want a hotel in Paris with a budget of \$200."
Later session: "Find me a hotel." → Bot recalls Paris and \$200

---

## Strengths

* Long-term memory persists across sessions
* Context-aware responses from LLM
* Modular design with SQLite-based memory store
* Simple, demo-ready Streamlit UI

## Weaknesses

* Basic user input parsing (budget/location via keywords)
* LLM occasionally ignores context
* Session memory isn’t persisted
* No encryption/authentication
* SQLite not suited for high concurrency

---

## Challenges Solved

* **JSONDecodeError:** Handled with error fallback and file init
* **Inconsistent memory use:** Refined prompts
* **Scalability:** Replaced JSON with SQLite
* **Schema design:** Used `data_type` field for flexibility

---

## Future Improvements

* NLP parsing of preferences (e.g., spaCy, regex)
* Replace SQLite with PostgreSQL/DynamoDB
* Persist session memory between app restarts
* Add encryption and user authentication
* Optimize performance (batch writes, cache)
* Expand hotel booking features (e.g., mock API integration)
* Enhance UI (reset button, memory viewer)
* Robust error handling/logging

---

## Summary

A working PoC demonstrating long-term context in a hotel booking chatbot using LangChain and Llama 3.1:8B. Migrated from JSON to SQLite for memory storage, with prompt engineering to generate memory-aware responses. Demonstrates key principles of persistent memory, modular architecture, and user-specific conversations, ready for enhancement into a production system.
