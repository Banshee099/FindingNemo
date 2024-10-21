import streamlit as st
import openai
import time
from datetime import datetime
from typing import List, Dict
from openai import OpenAI





class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: datetime = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()


def get_llm_response(messages: List[Dict[str, str]]) -> str:
    """Get response from OpenAI with error handling and retries"""

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )

    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[{"role": "user",
                   "content": f"""{messages}"""}],
        temperature=0.5,
        top_p=1,
        max_tokens=2048,
        stream=False
    )
    return completion.choices[0].message.content


def initialize_session_state():
    """Initialize session state variables"""
    if "user_sessions" not in st.session_state:
        st.session_state.user_sessions = {}
    if "current_user" not in st.session_state:
        st.session_state.current_user = None




st.set_page_config(
    page_title="Multi-User Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
initialize_session_state()

st.title("💬 Multi-User Chatbot")

# Sidebar for user management
with st.sidebar:
    st.header("User Management")
    username = st.text_input("Enter username")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if username:
                st.session_state.current_user = username
                if username not in st.session_state.user_sessions:
                    st.session_state.user_sessions[username] = []
                st.success(f"Logged in as {username}")
            else:
                st.error("Please enter a username")

    with col2:
        if st.button("Logout"):
            st.session_state.current_user = None
            st.success("Logged out successfully")
            st.rerun()

# Main chat interface
if st.session_state.current_user:
    st.write(f"Logged in as: {st.session_state.current_user}")

    # Chat messages container with custom styling
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.user_sessions[st.session_state.current_user]:
            with st.chat_message(message.role):
                st.write(message.content)
                st.caption(f"Sent at: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        try:
            # Add user message
            user_message = ChatMessage("user", prompt)
            st.session_state.user_sessions[st.session_state.current_user].append(user_message)

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Get and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                # Prepare conversation history
                formatted_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in st.session_state.user_sessions[st.session_state.current_user]
                ]

                try:
                    response = get_llm_response(formatted_messages)

                    # Simulate typing effect
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.write(full_response + "▌")
                    message_placeholder.write(response)

                    # Add assistant response to history
                    assistant_message = ChatMessage("assistant", response)
                    st.session_state.user_sessions[st.session_state.current_user].append(assistant_message)

                except Exception as e:
                    st.error("Failed to get response. Please try again.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Session management in sidebar
    with st.sidebar:
        st.header("Session Management")

        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.user_sessions[st.session_state.current_user] = []
            st.success("Chat history cleared!")
            st.rerun()

        # Display session info
        st.header("Session Info")
        messages = st.session_state.user_sessions[st.session_state.current_user]
        st.write(f"Total messages: {len(messages)}")
        if messages:
            st.write(f"Session started: {messages[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Last message: {messages[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.info("Please login using the sidebar to start chatting.")