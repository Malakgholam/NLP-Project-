# app.py
import streamlit as st
from app import ask_question  # ✅ Import your ask_question function

# 🎨 Streamlit Page Config
st.set_page_config(page_title="NLP BUDDY", page_icon="🤖", layout="centered")

st.title("🤖 NLP BUDDY")
st.write("Ask me any NLP-related question and I will answer from your knowledge base!")

# ✅ Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat display
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**🧑 You:** {message['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {message['content']}")
    st.markdown("---")

# User input
query = st.text_input("💬 Enter your question:", key="user_input")

if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        answer, summary, pages = ask_question(query)

    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Append bot response
    if summary:
        response_text = f"📝 **Summary**\n{summary}"
    else:
        response_text = f"✅ **Answer**\n{answer}"

    # Add page numbers if available
    if pages:
        response_text += f"\n\n📄 **Source Pages:** {', '.join(map(str, pages))}"

    st.session_state.chat_history.append({"role": "bot", "content": response_text})

    st.rerun()
