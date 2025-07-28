# Task 4: RAG-Powered Chatbot with Gemini and FAISS

## 📋 Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot system using Google's Gemini AI model and FAISS vector database. The system provides context-aware responses by retrieving relevant information from a custom knowledge base and generating intelligent responses.

### 🎯 Objectives
- Build a context-aware chatbot using RAG architecture
- Implement document retrieval using FAISS vector database
- Integrate Google Gemini AI for natural language generation
- Create an interactive Streamlit web interface
- Demonstrate practical application of RAG systems

## 🏗️ Architecture & Methodology

### 1. RAG System Components

#### A. Document Processing
- **Text Loading**: Load custom knowledge base documents
- **Text Splitting**: Chunk documents into manageable segments
- **Embedding Generation**: Convert text chunks to vector embeddings

#### B. Vector Database (FAISS)
- **Index Creation**: Build efficient search index
- **Similarity Search**: Retrieve relevant document chunks
- **Local Storage**: Persist vector database for reuse

#### C. Language Model (Gemini)
- **Context Integration**: Combine retrieved context with user queries
- **Response Generation**: Generate coherent and relevant responses
- **Conversation Memory**: Maintain chat history for context

### 2. Technology Stack
- **LLM**: Google Gemini Pro (2.5-pro)
- **Embeddings**: Google Generative AI Embeddings
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Web Framework**: Streamlit
- **Language**: Python with LangChain

## 🚀 Setup & Installation

### Prerequisites
```bash
Python 3.8+
Google Gemini API Key
```

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Task4
   ```

2. **Install required packages**
   ```bash
   pip install langchain langchain-community faiss-cpu streamlit google-generativeai langchain-google-genai
   ```

3. **Set up Google API Key**
   ```python
   import os
   os.environ["GOOGLE_API_KEY"] = "your_google_api_key_here"
   ```

4. **Prepare your knowledge base**
   - Place your text documents in the project directory
   - Update the file path in the notebook: `loader = TextLoader("your_document.txt")`

## 📊 Implementation Details

### 1. Document Processing Pipeline

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader("Machine_learning.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)
```

### 2. Vector Database Setup

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Build vector store
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index_gemini")
```

### 3. RAG Chain Configuration

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)
```

## 🌐 Streamlit Web Application

### Running the App

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`
   - The chatbot interface will be available

### App Features

- **Interactive Chat Interface**: Real-time conversation with the RAG system
- **Context-Aware Responses**: Responses based on retrieved document chunks
- **Conversation Memory**: Maintains chat history for context
- **Error Handling**: Graceful error handling and user feedback
- **Responsive Design**: Clean and intuitive user interface

### App Structure

```python
# Main Streamlit app components
st.set_page_config(page_title="Gemini Chatbot 💬", layout="centered")
st.title("🤖 Gemini-Powered Context-Aware Chatbot")

# Chat interface
user_input = st.text_input("You:", key="input")
if user_input:
    result = qa_chain({"question": user_input})
    # Display response
```

## 🔍 Key Features

### 1. Document Retrieval
- **Semantic Search**: Find relevant document chunks using embeddings
- **Context Integration**: Combine retrieved context with user queries
- **Relevance Scoring**: Rank retrieved documents by similarity

### 2. Natural Language Generation
- **Context-Aware Responses**: Generate responses based on retrieved information
- **Conversation Flow**: Maintain coherent conversation threads
- **Temperature Control**: Adjust response creativity and consistency

### 3. Memory Management
- **Conversation History**: Store and retrieve previous interactions
- **Context Persistence**: Maintain context across multiple turns
- **Session Management**: Handle user sessions efficiently

## 🛠️ Usage Examples

### Basic RAG Query
```python
# Simple question-answering
question = "What is machine learning?"
result = qa_chain({"question": question})
print(result["answer"])
```

### Interactive Chat
```python
# Multi-turn conversation
chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    result = qa_chain({"question": user_input})
    print(f"Bot: {result['answer']}")
```

### Document Processing
```python
# Process new documents
def add_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    vectorstore.add_documents(docs)
```

## 🔧 Customization

### Adding New Documents
1. Place your text files in the project directory
2. Update the file path in the notebook
3. Re-run the document processing pipeline
4. Rebuild the FAISS index

### Modifying the Model
1. **Change LLM**: Update the model parameter in `ChatGoogleGenerativeAI`
2. **Adjust Temperature**: Modify creativity level (0.0-1.0)
3. **Update Embeddings**: Change embedding model for different domains

### Customizing the Interface
1. **UI Styling**: Modify Streamlit components in `app.py`
2. **Response Format**: Customize response generation in the chain
3. **Memory Settings**: Adjust conversation memory parameters

## 📈 Performance Optimization

### 1. Chunk Size Optimization
- **Smaller Chunks**: Better precision, more context switching
- **Larger Chunks**: More context, potential information loss
- **Overlap**: Balance between continuity and efficiency

### 2. Retrieval Parameters
- **Top-k Retrieval**: Number of document chunks to retrieve
- **Similarity Threshold**: Minimum similarity score for retrieval
- **Reranking**: Additional ranking of retrieved documents

### 3. Model Configuration
- **Temperature**: Lower for factual responses, higher for creative ones
- **Max Tokens**: Control response length
- **Top-p Sampling**: Adjust response diversity

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```python
   # Ensure API key is set correctly
   os.environ["GOOGLE_API_KEY"] = "your_actual_api_key"
   ```

2. **FAISS Index Not Found**
   ```python
   # Rebuild the index
   vectorstore = FAISS.from_documents(docs, embeddings)
   vectorstore.save_local("faiss_index_gemini")
   ```

3. **Memory Issues**
   ```python
   # Clear conversation memory
   memory.clear()
   ```

4. **Streamlit Connection Error**
   ```bash
   # Restart Streamlit
   streamlit run app.py --server.port 8501
   ```

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables
- Never commit API keys to version control
- Use secure key management services in production

### Data Privacy
- Ensure documents don't contain sensitive information
- Implement user authentication for production use
- Consider data retention policies

## 📊 Evaluation Metrics

### Response Quality
- **Relevance**: How well responses match user queries
- **Accuracy**: Factual correctness of generated responses
- **Coherence**: Logical flow and readability

### System Performance
- **Response Time**: Speed of query processing
- **Retrieval Quality**: Relevance of retrieved documents
- **Memory Usage**: Efficient resource utilization
