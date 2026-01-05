import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"



# Page config
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Q&A Assistant")
st.caption("Upload a PDF and get precise answers extracted directly from the document")


# Load model 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm():
    model_name = "Qwen/Qwen3-4B"  # use instruct variant for QA
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # FP16 for faster GPU inference
        device_map="auto",
        trust_remote_code=True
    )
    
    # Compile the model for faster generation
    model = torch.compile(model)
    
    return tokenizer, model

# Load model
tokenizer, model = load_llm()




def generate_text(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# PDF processing
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(chunks, embedding)
    return vectordb

def ask_question(vectordb, query):
    docs = vectordb.similarity_search(query, k=1)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a question-answering assistant.
Answer using only the provided context.
If the answer is not in the context, say "I don't know".Write a detailed answer of approximately 400 words.
Do not conclude early.

Context:
{context}

Question: {query}
Answer:
"""
    answer = generate_text(prompt)
    return answer.split("Answer:")[-1].strip()


# UI
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        vectordb = process_pdf(pdf_path)

    st.success("PDF processed successfully")

    question = st.text_input("Ask a question")

    if question:
        with st.spinner("Generating answer..."):
            answer = ask_question(vectordb, question)

        st.markdown("### Answer")
        st.write(answer)

