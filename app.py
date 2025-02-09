import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GROQ_API_KEY")


# Initialize the ChatGroq model
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=api_key)

# Streamlit UI
st.title("📄 AI-Powered Cover Letter Generator")
st.write("Upload your **resume (PDF)** and enter a **job description** to generate a professional cover letter.")

# File uploader for resume PDF
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Enter the Job Description")

# Generate button
if uploaded_file and job_description:
    # Save uploaded file temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load the resume PDF
    loader = PyPDFLoader("temp_resume.pdf")
    docs = loader.load()

    # Extract text from all pages
    resume = "\n".join([doc.page_content for doc in docs])

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Looking at the job description below, write a professional cover letter based on the resume:

    **Job Description:**
    {job_description}

    **Resume:**
    {resume}
    """)

    # Define the chain
    chain = (
        prompt 
        | chat 
        | StrOutputParser()  # Ensures output is a string
    )

    # Invoke the chain
    with st.spinner("Generating your cover letter... ⏳"):
        cover_letter = chain.invoke({"job_description": job_description, "resume": resume})

    # Display the cover letter
    st.subheader("📜 Generated Cover Letter")
    st.text_area("Your AI-generated cover letter:", cover_letter, height=400)

    # Download button for the generated cover letter
    st.download_button(
        label="📥 Download Cover Letter",
        data=cover_letter,
        file_name="cover_letter.txt",
        mime="text/plain",
    )
