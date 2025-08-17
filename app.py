import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# --- PDF processing ---
def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid NoneType errors
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into overlapping chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

# --- Vector Store (FAISS) ---
def get_vector_store(chunks):
    """Generate FAISS vector store from text chunks and save locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# --- QA Chain ---
def get_conversational_chain():
    """Create a question-answering chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible using only the provided context.
    If the answer is not in the context, just say "answer is not available in the context."
    Do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        client=genai,
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# --- Chat Helpers ---
def clear_chat_history():
    """Clear stored messages in session."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]

def user_input(user_question):
    """Retrieve FAISS vector store and generate a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")  # type: ignore

    # Allow deserialization since we're loading our own index
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    return response

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    # Sidebar PDF uploader
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Submit & Process'",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text could be extracted from the PDFs.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")

    # Main chat area
    st.title("ChatMulti-PDF 🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate and display bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = response.get("output_text", "")
                st.markdown(full_response)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
