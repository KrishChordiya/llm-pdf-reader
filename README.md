# ChatPDF: LLM-Powered PDF Q&A with Streamlit and Gemini

This project is a web application built with Streamlit that allows users to upload PDF documents and ask questions about their content. The application uses LangChain and Google's Gemini models (via `langchain-google-genai`) to understand the documents and generate relevant answers based *solely* on the provided text (Retrieval-Augmented Generation - RAG).

![Screenshot From 2025-04-02 06-08-05](https://github.com/user-attachments/assets/8755cf83-078a-442e-b13e-d567c4e2dc2e)

## Live Application

You can access the live version of this application here:

**➡️ [https://llm-pdf-reader.streamlit.app/] ⬅️**

## Features

*   **Upload Multiple PDFs:** Users can upload one or more PDF files simultaneously.
*   **Text Extraction:** Extracts text content directly from the uploaded PDFs using `PyPDF2`.
*   **Text Chunking:** Splits the extracted text into manageable chunks using `langchain.text_splitter`.
*   **Vector Embeddings:** Generates vector embeddings for text chunks using Google's `text-embedding-004` model.
*   **Vector Store:** Stores and indexes embeddings efficiently using `FAISS` for fast retrieval.
*   **Conversational Q&A:** Provides a chat interface (powered by Streamlit's `st.chat_input` and `st.chat_message`) to ask questions about the documents.
*   **Context-Aware Answers:** Uses Google's Gemini model (`gemini-1.5-flash`) via a LangChain retrieval chain to generate answers based *only* on the relevant context retrieved from the uploaded documents.
*   **Source Viewing:** Optionally allows users to view the specific text chunks (sources) used by the LLM to generate the answer.
*   **State Management:** Uses `st.session_state` to maintain the vector store and chat history across user interactions.

## Technologies Used

*   **Python 3.x**
*   **Streamlit:** For building the interactive web application UI.
*   **LangChain:** Framework for developing applications powered by language models (text splitting, vector stores, retrieval chains, prompts).
*   **langchain-google-genai:** Integration for using Google's Generative AI models (Gemini and Embeddings) within LangChain.
*   **google-generativeai:** The underlying Google AI Python SDK.
*   **FAISS (faiss-cpu):** Library for efficient similarity search and clustering of dense vectors.
*   **PyPDF2:** Library for reading and extracting text from PDF files.
*   **python-dotenv:** For managing environment variables (like API keys) locally.

## Usage

1.  **Upload Documents:** Use the sidebar file uploader to select one or more PDF files.
2.  **Process Documents:** Click the "Process" button in the sidebar. The application will extract text, create chunks, generate embeddings, and build the vector store. A success message will appear upon completion.
3.  **Ask Questions:** Type your questions about the content of the uploaded PDFs into the chat input box at the bottom of the main page and press Enter.
4.  **View Answers:** The AI's answer, based on the document context, will appear in the chat window.
5.  **View Sources (Optional):** If you want to see which parts of the documents were used to generate the answer, toggle the "View Sources" switch at the top. The relevant text snippets will be displayed below the assistant's message.

## Code Explanation (`app.py`)

*   **`get_pdf_text(pdf_docs)`:** Takes a list of uploaded PDF files, reads each one using `PyPDF2`, extracts text from each page, and concatenates it into a single string.
*   **`get_text_chunks(text)`:** Takes the combined text and splits it into smaller chunks using `CharacterTextSplitter` for easier processing by the LLM.
*   **`get_vectorstore(text_chunks)`:** Generates embeddings for the text chunks using `GoogleGenerativeAIEmbeddings` and creates a `FAISS` vector store from these embeddings. This vector store is stored in `st.session_state.vectorstore`.
*   **`get_retrival_chain(query)`:**
    *   Retrieves the vector store from the session state.
    *   Initializes the `ChatGoogleGenerativeAI` model (Gemini).
    *   Defines a `ChatPromptTemplate` instructing the LLM to answer based *only* on the provided context.
    *   Creates a `stuff` documents chain (`create_stuff_documents_chain`) to combine retrieved documents into the prompt.
    *   Creates a `retrieval_chain` (`create_retrieval_chain`) that first retrieves relevant documents from the FAISS vector store based on the user `query` (fetching top `k=3` documents) and then passes them to the `stuff` chain for answer generation.
    *   Invokes the chain with the user query and returns the response dictionary (containing the answer and the source context documents).
*   **`main()`:**
    *   Loads environment variables (primarily for local execution).
    *   Sets up the Streamlit UI elements: chat input, source toggle, sidebar for file upload and processing.
    *   Manages chat history using `st.session_state.messages`.
    *   Displays past chat messages. If "View Sources" is toggled, it displays the context documents associated with assistant messages.
    *   Handles user input:
        *   Checks if documents have been processed (`vectorstore` exists).
        *   Adds the user message to the chat history.
        *   Calls `get_retrival_chain` to get the AI response (showing a spinner during processing).
        *   Adds the AI response (answer and context) to the chat history.
        *   Uses `st.rerun()` to immediately update the UI with the new messages.
    *   Handles the sidebar logic:
        *   File uploader allows selecting PDF files.
        *   "Process" button triggers the document processing workflow (`get_pdf_text`, `get_text_chunks`, `get_vectorstore`) when files are uploaded, showing a spinner during processing.
