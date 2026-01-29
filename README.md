Ask My Documents

A Streamlit chatbot that allows you to chat with your own documents
(PDF, DOCX, TXT) using Groq LLaMA 3.1, LangChain, and FAISS.

<img width="1788" height="870" alt="image" src="https://github.com/user-attachments/assets/66a61be9-5693-4797-8c70-47902c231442" />


FEATURES
--------
- Chat with multiple documents
- Supports PDF, DOCX, and TXT files
- Conversational memory
- Source references for answers
- Fast responses powered by Groq

TECH STACK
----------
- Streamlit (UI)
- LangChain (RAG pipeline)
- FAISS (Vector database)
- HuggingFace Embeddings
- Groq LLaMA 3.1 (LLM)


PROJECT STRUCTURE
-----------------
.   
|-- app.py   
|-- data/              (place your documents here)   
|-- .env               (GROQ_API_KEY)   
|-- requirements.txt   
|-- README.txt   


SETUP INSTRUCTIONS
------------------

1. Clone the repository
   git clone https://github.com/SedulousN/QA_CHATBOT.git   
   cd QA_CHATBOT

3. Install dependencies
   pip install -r requirements.txt

4. Add your Groq API key
   Create a file named .env and add:  
   GROQ_API_KEY=your_groq_api_key_here

5. Add documents
   Place your files inside the data/ folder:
   - PDF files
   - DOCX files
   - TXT files

6. Run the application
   streamlit run app.py


USAGE
-----
- Ask questions in natural language
- The chatbot answers only from your documents
- View which documents were used for each answer


FUTURE IMPROVEMENTS
-------------------
- Streaming responses
- Highlight text in source documents
- Light/Dark mode toggle
- Upload files from the UI


ACKNOWLEDGEMENTS
----------------
LangChain
Groq
HuggingFace
Streamlit
