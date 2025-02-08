from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load document (Fixed path issue)
loader = TextLoader('./data/history.txt')  # Ensure the path is correct
history_doc = loader.load()

# Debugging: Check if the document is loaded properly
print("Loaded Document:", history_doc)

# Split document with adjusted chunk size and overlap
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
history_document = text_splitter.split_documents(history_doc)

# Debugging: Check if the document is split correctly
print(f"Total Document Chunks: {len(history_document)}")

# Initialize Sentence Transformer Model
embedding_function = SentenceTransformer('all-MiniLM-L6-v2')

# Store in ChromaDB
db = Chroma.from_documents(history_document, embedding_function)

# Debugging: Check if ChromaDB has stored the documents
print("Number of Documents in DB:", db._collection.count())

# Perform similarity search
query = "cornerstone of modern AI"
similar_docs = db.similarity_search(query)

# Output results
if similar_docs:
    print("Similar Documents Found:")
    for doc in similar_docs:
        print(doc.page_content)
else:
    print("No relevant documents found.")
