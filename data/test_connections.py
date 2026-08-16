"""
Testing connection to all apis
"""

import os 
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        print("open ai connected:", response.choices[0].message.content)
        return True
    except Exception as e:
        print(f" opent ai failed {e}")
        return False

def test_gemini():
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model = genai.GenerativeModel("gemini-3.5-flash-lite")
        response = model.generate_content("Say OK in one word")

        print(f"gemini connected : {response.text.strip}")
        return True
    except Exception as e:
        print(f" gemini failed {e}")
        return False

def test_gemini_embeddings():
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content="Test embedding for Docusense ai",
            task_type="retrieval_document"
        )

        embedding = result['embedding']
        print(f" gemini embeddings: vector size = {len(embedding)}")
        return True
    except Exception as e:
        print(f"embedding failed {e}")
        return False

def test_huggingface():
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        user = api.whoami()
        print(f"hugging face connected: {user['name']}")
        return True
    except Exception as e:
        print(f"hugging face failed:{e}")
        return False

def test_qdrant():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url= os.getenv("QDRANT_URL"),
            api_key= os.getenv("QDRANT_API_KEY")
        )
        collections = client.get_collections()
        print(f" Qdrant connected: {len(collections.collections)} collections")
        return True
    except Exception as e:
        print(f"qdrant failed: {e}")
        return False

def test_langsmith():
    try:
        from langsmith import Client
        client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
        projects = list(client.list_projects())
        print(f"langsmith is connected : {len(projects)} projects")
        return True
    except Exception as e:
        print(f"langsmith failed {e}")
        return False

def test_documents():
    from pathlib import Path
    import json

    raw_dir = Path("data/raw")
    meta_file = Path("data/document_metadata.json")

    pdfs = list(raw_dir.glob("*.pdf"))
    print(f"Documents found: {len(pdfs)} pdfs")

    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"metadatafile: {len(meta)} documents tracked")

    return len(pdfs) > 0


def test_langchain_gemini():
    """test langchaing + gemini integration"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key= os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        response = llm.invoke("Say Ready in one word")
        print(f"langchain gemining working : {response.content.strip()}")
        return True
    except Exception as e:
        print(f"langchain gemini failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*50)
    print("docusense ai connect testing")
    print("="*50)

    results = {
        # "huggingface": test_huggingface(),
        # "qdrant": test_qdrant(),
        # "langsmith": test_langsmith(),
        # "documents": test_documents(),
        # "gemini": test_gemini(),
        "embedding": test_gemini_embeddings(),
        #"langchain": test_langchain_gemini(),
    }

    print("\n" + "="*50)
    passed = sum(results.values())
    total = len(results)
    print(f" Results: {passed}/{total} passed")

    if passed == total:
        print("all connection are ready to go yahooooo....")
    else:
        failed = [k for k,v in results.items() if not v]
        print(f" need to fix these to go next step try again.......")

    print("\n" + "="*50)
