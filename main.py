import streamlit as st
from dataclasses import dataclass
from typing import List
import datetime
from openai import OpenAI



@dataclass
class Message:
    """Class for storing chat messages."""
    content: str
    role: str
    timestamp: datetime.datetime


class ChatMemory:
    """Class for managing chat history and context."""

    def __init__(self, max_context_length: int = 1000):
        self.messages: List[Message] = []
        self.max_context_length = max_context_length

    def add_message(self, content: str, role: str):
        """Add a new message to the chat history."""
        message = Message(
            content=content,
            role=role,
            timestamp=datetime.datetime.now()
        )
        self.messages.append(message)

    def get_context(self) -> str:
        """Get formatted context string from chat history."""
        context = ""
        for msg in self.messages[-5:]:  # Get last 5 messages for context
            context += f"{msg.role}: {msg.content}\n"
        return context.strip()

    def clear(self):
        """Clear chat history."""
        self.messages = []


class Chatbot:
    """Main chatbot class handling model interactions."""

    def __init__(self):
        self.client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1",
                             api_key = "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"
                             )
        self.memory = ChatMemory()

    def generate_response(self, user_input: str) -> str:
        """Generate response using the model."""
        # Get context from previous messages
        context = self.memory.get_context()

        # Combine context and new input
        prompt = f"{context}\nUser: {user_input}\nAssistant:"

        # Tokenize and generate response
        completion = self.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": f"""{prompt}"""}],
            temperature=0.5,
            top_p=1,
            max_tokens=2048,
            stream=False
        )

        # Decode response
        response = completion.choices[0].message.content

        # Add to memory
        self.memory.add_message(user_input, "User")
        self.memory.add_message(response, "Assistant")

        return response


def initialize_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Chatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def main():
    st.title("Contextual Chatbot")

    # Initialize session state
    initialize_session_state()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.generate_response(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar controls
    with st.sidebar:
        st.title("Settings")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chatbot.memory.clear()
            st.rerun()

main()