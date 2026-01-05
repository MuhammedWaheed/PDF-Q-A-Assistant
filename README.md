# 📄 PDF Q&A Assistant

A powerful question-answering system that extracts precise answers from PDF documents using advanced AI models. Built with Streamlit, LangChain, and the Qwen3-4B language model.

## 🌟 Features

- **PDF Upload**: Upload any PDF document for analysis
- **Intelligent Q&A**: Ask questions and get accurate answers extracted directly from the document
- **Vector Search**: Uses FAISS for efficient semantic search across document chunks
- **GPU-Accelerated**: Leverages GPU for faster inference with the Qwen3-4B model
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **Context-Aware Responses**: Generates detailed answers (approximately 400 words) based on relevant document context

## 🏗️ Architecture

The system uses a Retrieval-Augmented Generation (RAG) approach:

1. **Document Processing**: PDFs are loaded and split into manageable chunks
2. **Embedding**: Text chunks are converted to vector embeddings using sentence-transformers
3. **Vector Storage**: FAISS indexes the embeddings for fast similarity search
4. **Query Processing**: User questions are matched with relevant document chunks
5. **Answer Generation**: Qwen3-4B generates contextual answers based on retrieved chunks

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Language Model**: Qwen3-4B (4 billion parameters)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Document Processing**: LangChain, PyPDF
- **Deep Learning Framework**: PyTorch
- **Tunneling**: Ngrok (for remote access)

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- 8GB+ RAM
- 10GB+ free disk space (for model downloads)

## 💻 Usage

### Running Locally

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a PDF file using the file uploader
   - Wait for the PDF to be processed
   - Enter your question in the text input
   - View the generated answer

### Running with Ngrok (Remote Access)

If you want to access the application remotely:

1. **Set up Ngrok authentication**
   - Sign up for a free account at [ngrok.com](https://ngrok.com/)
   - Get your auth token from the dashboard
   - Update the auth token in the code or set it via command line:
     ```python
     from pyngrok import ngrok
     ngrok.set_auth_token("YOUR_AUTH_TOKEN")
     ```

2. **Start the tunnel**
   ```bash
   python
   >>> from pyngrok import ngrok
   >>> ngrok.connect(8501)
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py --server.port 8501
   ```

4. **Access via the ngrok URL**
   - The tunnel will provide a public URL (e.g., `https://xxxxx.ngrok-free.dev`)
   - Share this URL to access the app from anywhere

## 📁 Project Structure

```
pdf-qa-assistant/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── pdf-q-a-assistant.ipynb  # Original Jupyter notebook
```

## 🎯 How It Works

1. **PDF Processing**:
   - The uploaded PDF is loaded using `PyPDFLoader`
   - Content is split into chunks (500 characters with 50 character overlap)
   - Each chunk is converted to embeddings using HuggingFace's all-MiniLM-L6-v2 model

2. **Vector Search**:
   - User's question is converted to an embedding
   - FAISS performs similarity search to find the most relevant chunk
   - Top-k (k=1) most similar chunks are retrieved

3. **Answer Generation**:
   - Retrieved context is passed to the Qwen3-4B model
   - The model generates a detailed answer based solely on the provided context
   - If the answer isn't in the context, the model responds with "I don't know"n


## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU instead: remove `device_map="auto"`
   - Use a smaller model variant

2. **Slow Performance**
   - Ensure GPU is being utilized
   - Check CUDA installation: `torch.cuda.is_available()`
   - Consider using quantized models


## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Qwen](https://huggingface.co/Qwen) for the language model
- [LangChain](https://langchain.com/) for the document processing framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io/) for the web framework
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library


