import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

from scrape import scrape_website, extract_body_content, clean_body_content
from parse import parse_with_ollama

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Page setup
st.set_page_config(page_title="Web Scraper + RAG Chatbot", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 AI Web Scraper + 🤖 RAG Chatbot")

# Layout with two columns
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🔍 Web Scraping")
    url = st.text_input("Enter the website URL", placeholder="https://example.com")

    if st.button("Scrape Site"):
        with st.spinner("Scraping website..."):
            dom_content = scrape_website(url)
            body_content = extract_body_content(dom_content)
            cleaned_content = clean_body_content(body_content)
            st.session_state.dom_content = cleaned_content
            st.success("Website scraped successfully!")

    if "dom_content" in st.session_state:
        with st.expander("📄 View Cleaned DOM Content", expanded=False):
            st.text_area("DOM Content", st.session_state.dom_content, height=300)

with col2:
    st.subheader("📄 Document & Chat Setup")
    session_id = st.text_input("Session ID", value="default_session", help="Unique ID for your conversation session")

    uploaded_files = ["rgukt.pdf"]  # Replace with uploader if needed
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            loader = PyPDFLoader(file)
            docs = loader.load()
            documents.extend(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever()

        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
                           "formulate a standalone question which can be understood without the chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for question-answering tasks. "
                           "Use the following pieces of retrieved context to answer the question. "
                           "If you don't know the answer, say that you don't know. Keep the answer concise.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            if 'store' not in st.session_state:
                st.session_state.store = {}

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            st.subheader("💬 Chat with AI")
            user_input = st.text_input("Your question:", placeholder="Ask something based on the PDF or scraped data...")

            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.success("Assistant's Response:")
                st.markdown(f"**{response['answer']}**")

        else:
            st.warning("Please set `GROQ_API_KEY` in the `.env` file.")

