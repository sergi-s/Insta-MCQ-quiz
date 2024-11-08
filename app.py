import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")


def get_pdf_text(pdf_docs):
    """Extract text from the uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split the extracted text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Create a vector store from the text chunks."""
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def save_vectorstore(vectorstore, directory="faiss_index"):
    """Save the FAISS vectorstore to a directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    vectorstore.save_local(directory)
    print(f"Vectorstore saved to {directory}")


def load_vectorstore(directory="faiss_index"):
    """Load the FAISS vectorstore from the directory."""
    if os.path.exists(directory):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.load_local(directory, embeddings=embeddings)
        print(f"Vectorstore loaded from {directory}")
        return vectorstore
    else:
        return None

from langchain.schema import SystemMessage, HumanMessage
import re


def generate_mcq_question(chunk, vectorstore):
    """Generate a multiple-choice question based on the text chunk."""

    # Search for the most relevant passage from the vectorstore
    similar_docs = vectorstore.similarity_search(chunk, k=3)
    context = "\n".join([doc.page_content for doc in similar_docs])

    question_prompt = (
        f"Generate a multiple-choice question (with 4 options and the correct answer) based on this text:\n\n{context}\n\n"
        f"Format:\n"
        f"Question: <question>\n"
        f"A. <option 1>\n"
        f"B. <option 2>\n"
        f"C. <option 3>\n"
        f"D. <option 4>\n"
        f"Correct Answer: <correct answer>\n"
        f"Ensure the correct answer is clearly marked."
    )

    # Initialize the LLM with LangChain and the loaded API key
    llm = ChatOpenAI(openai_api_key=api_key)

    # Create a list of messages following LangChain's expected format
    messages = [HumanMessage(content=question_prompt)]
    
    try:
        # Use .call() to get the response directly from the model
        response = llm(messages)
        generated_content = response.content.strip()
        
        print("generated_content")
        print(generated_content)
        print("Sergiiiiii")

        # Regex to extract question, options, and answer
        pattern = r"Question:\s*(.*?)\nA\.\s*(.*?)\nB\.\s*(.*?)\nC\.\s*(.*?)\nD\.\s*(.*?)\nCorrect Answer:\s*(.*)"
        match = re.search(pattern, generated_content, re.DOTALL)
        
        print("match")
        print(match)

        if match:
            question = match.group(1).strip()
            print("question")
            print(question)
            
            options = [match.group(i).strip() for i in range(2, 6)]
            # options = {
            #     'A': match.group(2).strip(),
            #     'B': match.group(3).strip(),
            #     'C': match.group(4).strip(),
            #     'D': match.group(5).strip()
            # }
            print("options")
            print(options)
            correct_answer = match.group(6).strip()
            print("correct_answer")
            print(correct_answer)

            return question, correct_answer[3:], options
        else:
            print("Error: Unable to parse the generated question format.")
            return None, None, None

    except Exception as e:
        print(f"Error generating the question: {e}")
        return None, None, None


def handle_userinput(user_answer, correct_answer):
    """Handles user's answer and shows the result."""
    print("Handling user input...")
    print(f"User input: {user_answer}")
    print(f"Correct answer: {correct_answer}")
    if user_answer == correct_answer:
        st.write("Correct! 😊")
    else:
        st.write(f"Oops! The correct answer was: {correct_answer}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_chunk" not in st.session_state:
        st.session_state.current_chunk = None

    st.header("Chat with multiple PDFs :books:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get the text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Load or create the vector store
                vectorstore = load_vectorstore()  # Try loading the vectorstore
                if vectorstore is None:
                    print("Creating a new vectorstore...")
                    vectorstore = get_vectorstore(text_chunks)  # Create new vector store if not found
                    save_vectorstore(vectorstore)  # Save the new vectorstore
                
                # Generate the first question based on the first chunk
                st.session_state.current_chunk = text_chunks[1]  # Start with the first chunk
                st.warning("st.session_state.current_chunk: " + st.session_state.current_chunk)
                question, correct_answer, options = generate_mcq_question(st.session_state.current_chunk, vectorstore)
                print("question")
                print(question)
                st.session_state.current_question = (question, correct_answer, options)

                st.write("Your documents have been processed successfully! :tada:")

    # Show the current question if available
    if st.session_state.current_question:
        question, correct_answer, options = st.session_state.current_question
        
        # Display the question
        st.write(question)
        
        # Display multiple choice options
        user_answer = st.radio("Select an option", options)
        
        # Handle user answer
        if st.button("Submit Answer"):
            handle_userinput(user_answer, correct_answer)
            
        if st.button("Next"):
            vectorstore = load_vectorstore()
            st.session_state.current_question = generate_mcq_question(st.session_state.current_chunk, vectorstore)

            


if __name__ == '__main__':
    main()
