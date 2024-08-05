# MindMATE : RAG_MentalHealth
RAG chatbot for mental health 

# MentalHealth
The Mental Health project develops an assistant system for addressing mental health and wellness inquiries by leveraging document embedding, vector search, and language model-based generation using the phi3 model.

## Setup

1. Clone the repository:
```
git clone 
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Download and install Ollama from [Ollama](https://ollama.com/)

4. Pull the phi3 model using Ollama:
```
ollama pull phi3
```

## Creating the Vector Database

1. Ensure you have the PDF file `Mental Health Handbook English.pdf` in the `data/` directory.

2. Run the script to create the vector database:
```
python create_vectordb.py
```
This will create two files in the `database/` directory:
- `pdf_sections_index.faiss`: The FAISS index file
- `pdf_sections_data.pkl`: The pickle file containing section data

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```
The application will be accessible in your web browser at `http://localhost:8501`.

## Usage

1. Type your mental health and wellness query into the text input box.
2. Press the "Get Answer" button.
3. The system will find relevant information and create a response.
4. The response will be shown, and you can click the "Show Context" section to view the text used to generate the response.

## Files

- `create_vectordb.py`: Script for generating the vector database from the PDF document.
- `app.py`: The primary Streamlit application for the Q&A system.
- `data/`: Directory holding the source PDF.
- `database/`: Directory for storing the vector database files.

## Note

Make sure you have enough disk space and computational resources to run the vector database creation and the Streamlit application. Performance may vary based on your hardware capabilities.

## Youtube Demo
https://youtu.be/ZzhxlFAm0ho


