import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
import re
import random
import time

# choose an embedding ("OpenAI" or "instructor-xl")
# embeddings_type = "instructor-xl"
embeddings_type = "OpenAI"
embeddings = None
#? Notes: 
#? the faiss_index folder is used to store the embeddings of the text chunks, so that I dont need to calculate the embeddings every time I run the code. But this is for static testing purposes, in a real world scenario, the embeddings should be calculated every time, because the text chunks will be different every time.
#? if you used an embeddings different than the one used to create the cashed faiss_index folder, you will get an error, because the embeddings are not the same. 


static_testing = False
api_key = os.getenv("OPENAI_API_KEY")

def get_random_substring(s, y):
    if y > len(s): 
        raise ValueError("Substring length cannot be greater than string length.")
    
    start_idx = random.randint(0, len(s) - y)
    return s[start_idx:start_idx + y]

def load_embeddings():
    """Load the embeddings based on the specified type."""
    global embeddings
    start_time = time.time()
    if embeddings_type == "OpenAI":
        embeddings = OpenAIEmbeddings()
    elif embeddings_type == "instructor-xl":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    else:
        ValueError("Invalid embeddings type. Please choose 'OpenAI' or 'instructor-xl'.")
    end_time = time.time()
    print(f"Loaded embedding model in {end_time - start_time} seconds.")
    
    
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
        chunk_size=500, #1000
        chunk_overlap=100, #200
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from the text chunks."""
    load_embeddings()
    print("number of chunks {} chunks sizes: {}".format(len(text_chunks), [len(chunk) for chunk in text_chunks]))
    start_time = time.time()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    end_time = time.time()
    print(f"Running the embedding took: {end_time - start_time:.4f} seconds")
    return vectorstore

def save_vectorstore(vectorstore, directory="faiss_index"):
    """Save the FAISS vectorstore to a directory."""
    if not static_testing:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)
    vectorstore.save_local(directory)
    print(f"Vectorstore saved to {directory}")

def load_vectorstore(directory="faiss_index"):
    # this is for static testing purposes (save money and time), in a real world scenario, the embeddings should be calculated every time, because the text chunks will be different every time.
    """Load the FAISS vectorstore from the directory."""
    if not static_testing:
        return
    if os.path.exists(directory):
        load_embeddings()
        vectorstore = FAISS.load_local(directory, embeddings=embeddings)
        print(f"Vectorstore loaded from {directory}")
        return vectorstore
    else:
        return None
        
def generate_mcq_question_embedding(chunk, vectorstore, max_retries=3, retry_delay=2):
    """Generate a multiple-choice question based on the text chunk using the embedding with retries."""
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

    llm = ChatOpenAI(openai_api_key=api_key, verbose=True)
    messages = [HumanMessage(content=question_prompt)]

    attempt = 0
    while attempt < max_retries:
        try:
            # Generate response from the model
            response = llm(messages)
            generated_content = response.content.strip()
            
            # Parse the response
            pattern = r"Question:\s*(.*?)\nA\.\s*(.*?)\nB\.\s*(.*?)\nC\.\s*(.*?)\nD\.\s*(.*?)\nCorrect Answer:\s*(.*)"
            match = re.search(pattern, generated_content, re.DOTALL)

            if match:
                question = match.group(1).strip()
                options = [match.group(i).strip() for i in range(2, 6)]
                correct_answer = match.group(6).strip()
                return question, correct_answer[3:], options
            else:
                print("Error: Unable to parse the generated question format.")
                return None, None, None

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Returning None.")
                return None, None, None

    # If we exit the loop, all attempts failed
    print("Failed to generate a valid question after multiple attempts.")
    return None, None, None


def generate_mcq_question_text(chunk):
    """Generate a multiple-choice question based on the text chunk without using the embedding."""
    question_prompt = (
        f"Generate a multiple-choice question (with 4 options and the correct answer) based on this text:\n\n{chunk}\n\n"
        f"Format:\n"
        f"Question: <question>\n"
        f"A. <option 1>\n"
        f"B. <option 2>\n"
        f"C. <option 3>\n"
        f"D. <option 4>\n"
        f"Correct Answer: <correct answer>\n"
        f"Ensure the correct answer is clearly marked."
    )

    llm = ChatOpenAI(openai_api_key=api_key, verbose=True)
    messages = [HumanMessage(content=question_prompt)]
    
    try:
        response = llm(messages)
        generated_content = response.content.strip()
        
        pattern = r"Question:\s*(.*?)\nA\.\s*(.*?)\nB\.\s*(.*?)\nC\.\s*(.*?)\nD\.\s*(.*?)\nCorrect Answer:\s*(.*)"
        match = re.search(pattern, generated_content, re.DOTALL)

        if match:
            question = match.group(1).strip()
            options = [match.group(i).strip() for i in range(2, 6)]
            correct_answer = match.group(6).strip()
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
        st.write("Correct! ")
    else:
        st.write(f"Oops! The correct answer was: {correct_answer}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Insta MCQ generator", page_icon=":books:")

    if "current_question_embedding" not in st.session_state:
        st.session_state.current_question_embedding = None
    if "current_question_text" not in st.session_state:
        st.session_state.current_question_text = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None

    st.header("Insta MCQ generator :books:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.text_chunks = text_chunks
                
                vectorstore = load_vectorstore()
                if vectorstore is None:
                    vectorstore = get_vectorstore(text_chunks)
                    save_vectorstore(vectorstore)
                    st.session_state.vectorstore = vectorstore
                    
                st.session_state.vectorstore = vectorstore
                
                current_chunk = text_chunks[random.randint(0, len(text_chunks)-1)]
                # random substring of length 50
                random_substring = get_random_substring(current_chunk, 50)

                print(random_substring)
                print(len(random_substring))
                st.write("GIVEN TEXT len:", len(random_substring))
                st.write("GIVEN TEXT: ", random_substring)
                question_embedding, correct_answer_embedding, options_embedding = generate_mcq_question_embedding(random_substring, vectorstore)
                question_text, correct_answer_text, options_text = generate_mcq_question_text(random_substring)
                
                st.session_state.current_question_embedding = (question_embedding, correct_answer_embedding, options_embedding)
                st.session_state.current_question_text = (question_text, correct_answer_text, options_text)

                st.write("Your documents have been processed successfully! :tada:")

    if st.session_state.current_question_embedding and st.session_state.current_question_text:
        question_embedding, correct_answer_embedding, options_embedding = st.session_state.current_question_embedding
        question_text, correct_answer_text, options_text = st.session_state.current_question_text
        
        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)

        # Embedding-based question in the first column
        with col1:
            st.write("Question generated using embedding:")
            st.write(question_embedding)
            user_answer_embedding = st.radio("Select an option", options_embedding)
            if st.button("Submit Answer (Embedding)"):
                handle_userinput(user_answer_embedding, correct_answer_embedding)
        
        # Text-based question in the second column
        with col2:
            st.write("Question generated using text:")
            st.write(question_text)
            user_answer_text = st.radio("Select an option", options_text)
            if st.button("Submit Answer (Text)"):
                handle_userinput(user_answer_text, correct_answer_text)
    
    if st.button("Next"):
            with st.spinner("Loading next question..."):
                current_chunk = random.choice(st.session_state.text_chunks)
                
                st.session_state.current_question_embedding = generate_mcq_question_embedding(current_chunk, st.session_state.vectorstore)
                st.session_state.current_question_text = generate_mcq_question_text(current_chunk)
                
                    
                if not st.session_state.current_question_embedding or not st.session_state.current_question_text: 
                    st.warning("Could not generate the question, trying again...")
                st.experimental_rerun()

if __name__ == '__main__':
    main()