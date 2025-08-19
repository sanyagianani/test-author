import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from google.adk.agents import Agent


PDF_FILE = "prd.pdf" 
COLLECTION_NAME = "pdf_sections"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# DEBUG: Print the first section stored in the DB on load
try:
    all_docs = collection.get()
    if all_docs['documents']:
        print("🧠 First document in ChromaDB:")
        print(all_docs['documents'][0][:500])  
    else:
        print("⚠️ ChromaDB is initialized but empty. No documents found.")
except Exception as e:
    print(f"❌ Error fetching from ChromaDB: {e}")


# === PDF Extraction and ChromaDB Storage ===

def extract_pdf_sections(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    pdf_sections = {}
    with fitz.open(file_path) as doc:
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            text = page.get_text()
            section_key = f"Section {page_number + 1}"
            pdf_sections[section_key] = text.strip()
    return pdf_sections

def store_in_chromadb(sections):
    print("Storing sections to ChromaDB...")

    documents = list(sections.values())
    ids = [f"doc-{i}" for i in range(len(documents))]
    metadatas = [{"section": section} for section in sections.keys()]

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    all_docs = collection.get()
    print(f"✅ Upserted {len(all_docs['ids'])} documents to collection '{COLLECTION_NAME}'.")
    print(f"First document snippet:\n{all_docs['documents'][0][:300]}")


# === Query function used by both Terminal and ADK ===

def query_pdf_sections(query: str) -> str:
    print(f"[Agent] Received query: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=["documents", "metadatas"]
    )

    print("[Agent] Raw query result:", results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        print("[Agent] No relevant documents found.")
        return "Sorry, I couldn't find any relevant information in the document."

    snippet = docs[0].replace("\n", " ").strip()
    snippet = snippet[:500] + ("..." if len(snippet) > 500 else "")
    section_name = metas[0].get("section", "Unknown section")

    return f"Top result from {section_name}:\n\n{snippet}"


# === ADK Agent Definition ===

root_agent = Agent(
    name="pdf_qa_agent",
    model="gemini-2.0-flash",
    description="Answers user questions by returning the top relevant section from a PDF stored in ChromaDB.",
    instruction="You answer user questions about the PDF content by returning the top relevant excerpt.",
    tools=[query_pdf_sections],
)

# === One-time Setup on First Run ===

def setup_pdf_storage():
    # Check if ChromaDB is empty (collection has 0 docs)
    result = collection.query(query_texts=["test"], n_results=1)
    if not result["documents"] or not result["documents"][0]:
        print("🔄 No data found in ChromaDB. Extracting and saving PDF sections...")
        sections = extract_pdf_sections(PDF_FILE)
        store_in_chromadb(sections)
    else:
        print("✅ ChromaDB already contains data. Skipping PDF extraction.")

# === Terminal Mode Execution ===

def terminal_query():
    query = input("\nEnter your question about the document:\n> ").strip()
    if not query:
        print("⚠️ Empty query entered. Exiting.")
        return
    answer = query_pdf_sections(query)
    print("\n" + answer)

# === MAIN ===

if __name__ == "__main__":
    setup_pdf_storage()
    terminal_query()
