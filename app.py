import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
import csv
from dotenv import load_dotenv
import uuid
from google.api_core import retry
from google.api_core.exceptions import DeadlineExceeded

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyCxrnvzO_lqOfh4UglbO3QvHjwttPxnE0k"))
MODEL_NAME = "gemini-1.5-pro-001"

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        content = ""
        for page in range(len(pdf_reader.pages)):
            content += pdf_reader.pages[page].extract_text() + "\n"
        return content
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return None

# Function to generate summary using Gemini API with retry
@retry.Retry(predicate=retry.if_exception_type(DeadlineExceeded), initial=1.0, maximum=60.0, multiplier=2.0, deadline=300.0)
def generate_summary(text):
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                candidate_count=1
            )
        )
        
        prompt = f"""
        You are an expert researcher skilled in academic paper analysis. Given the following research paper content, provide a structured summary with the following sections: Objectives, Methods, and Key Findings. Format the response as a concise summary suitable for CSV output.

        Research Paper Content:
        {text[:15000]}  # Reduced to 15,000 characters to avoid timeout

        Provide the response in the following format:
        Objectives: [Summarized objectives]
        Methods: [Summarized methods]
        Key Findings: [Summarized key findings]
        """
        
        response = model.generate_content(prompt, request_options={"timeout": 300})  # Increased timeout to 300 seconds
        return response.text
    except DeadlineExceeded as e:
        st.error("API request timed out. Please try again or reduce the PDF size.")
        raise
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Function to parse summary and save to CSV
def save_to_csv(summary, output_file):
    try:
        lines = summary.split("\n")
        data = {}
        current_key = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Objectives:"):
                current_key = "Objectives"
                data[current_key] = line.replace("Objectives:", "").strip()
            elif line.startswith("Methods:"):
                current_key = "Methods"
                data[current_key] = line.replace("Methods:", "").strip()
            elif line.startswith("Key Findings:"):
                current_key = "Key Findings"
                data[current_key] = line.replace("Key Findings:", "").strip()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Objectives', 'Methods', 'Key Findings']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
        
        return output_file
    except Exception as e:
        st.error(f"Error saving CSV: {str(e)}")
        return None

# Streamlit App
st.set_page_config(page_title="Research Paper Summarizer", layout="wide")
st.title("Research Paper Summarizer with Gemini AI")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Research Paper")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Extract text from PDF
        pdf_text = extract_pdf_text(uploaded_file)
        if pdf_text:
            st.success("PDF Uploaded and Text Extracted Successfully!")
            
            # Preview PDF content
            st.header("PDF Content Preview")
            st.text_area("Preview", pdf_text[:1000] + "...", height=200)
            
            # Generate summary
            with st.spinner("Generating Summary with Gemini AI..."):
                summary = generate_summary(pdf_text)
                if summary:
                    # Save summary to CSV
                    output_file = f"summary_{uuid.uuid4()}.csv"
                    csv_file = save_to_csv(summary, output_file)
                    
                    if csv_file:
                        # Display summary
                        st.header("Generated Summary")
                        st.write(summary)
                        
                        # Provide download link for CSV
                        with open(csv_file, 'rb') as f:
                            st.download_button(
                                label="Download Summary as CSV",
                                data=f,
                                file_name=output_file,
                                mime="text/csv"
                            )
        else:
            st.error("Failed to process PDF. Please try another file.")
else:
    st.info("Please upload a research paper PDF to start.")
