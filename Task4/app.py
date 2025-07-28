import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI  # ✅ Use Chat class, not the basic one
)

# 🔐 Set Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCYD0BvwLra2MKBHyizyavRNsOu75fujjQ"

# 🧠 Load Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 📦 Load FAISS vector index
vectorstore = FAISS.load_local(
    "faiss_index_gemini",
    embeddings,
    allow_dangerous_deserialization=True  # ✅ Make sure it's safe
)

# 🤖 Load Gemini Chat Model (v2.5-pro is supported per your list)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)

# 💬 Enable conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔗 Set up conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 🌐 Streamlit UI setup
st.set_page_config(page_title="Gemini Chatbot 💬", layout="centered")
st.title("🤖 Gemini-Powered Context-Aware Chatbot")

# 📜 Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🧠 User input
user_input = st.text_input("You:", key="input")

if user_input:
    try:
        result = qa_chain({"question": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result["answer"]))
    except Exception as e:
        st.error(f"❌ Error: {e}")

# 💬 Show chat history
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
