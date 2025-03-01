import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=st.secrets["GOOGLE_API_KEY"])
    faiss_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return faiss_store

def get_retrival_chain(query):
    vectorstore = st.session_state.vectorstore
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

    retrieval_qa_chat_prompt = ChatPromptTemplate([
        ("system", "Answer any use questions based solely on the context below:\n<context>\n{context}\n</context>"),
        ("human","{input}")
    ])
    chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k':3}), chain)
    response = retrival_chain.invoke({"input":query})
    return response

def main():
    load_dotenv()

    user_input = st.chat_input("Your Question")
    toggle_sources = st.toggle("View Sources")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "context" in message and toggle_sources:
                 with st.expander("View Sources", expanded=False):
                    for doc in message["context"]:
                        st.write(f"📄 {doc.page_content[:300]}...")
                        st.divider()

    if user_input:
        if "vectorstore" not in st.session_state:
            st.error("No documents uploaded")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Show spinner while processing
            with st.spinner("Searching Documents..."):
                ai_response = get_retrival_chain(user_input)
                # Add AI response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_response["answer"], 
                    "context": ai_response["context"]
                })
            
            # Force immediate UI update
            st.rerun()

    with st.sidebar:
        files = st.file_uploader("Upload Documents 📂", accept_multiple_files=True, type=["pdf"])
        process_button = st.button("Process") 
        if process_button:
            if not files:
                st.warning("Upload documents first!")
            else:
                with st.spinner("Processing documents..."):  # Move spinner here
                    text = get_pdf_text(files)
                    text_chunks = get_text_chunks(text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.success("Document Processed")
                

if __name__=="__main__":
    main()
