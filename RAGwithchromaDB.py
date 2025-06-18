import google.generativeai as genai
from dotenv import load_dotenv
import os
import chromadb # Import Chroma
import uuid # To generate unique IDs for documents

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

    # --- 0. Configuration ---
    embedding_model_name = "text-embedding-004"
    generative_model_name = 'gemini-1.5-flash-latest'
    collection_name = "rag_eiffel_japan_collection" # Name for our Chroma collection

    # --- 1. Our "Knowledge Base" ---
    documents_kb = [ # Renamed to avoid conflict with chromadb 'documents' parameter
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
        "Constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design.",
        "The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and is the tallest structure in Paris.",
        "Millions of people ascend it every year, making it one of the most visited paid monuments in the world.",
        "The official currency of Japan is the Yen.",
        "Japan is an island country in East Asia, located in the northwest Pacific Ocean."
    ]
    # Create simple metadata (could be more complex, e.g., source, chapter)
    metadatas_kb = [{"doc_id": f"doc_{i+1}", "topic": "eiffel" if "eiffel" in doc.lower() else ("japan" if "japan" in doc.lower() else "other")} for i, doc in enumerate(documents_kb)]
    ids_kb = [str(uuid.uuid4()) for _ in documents_kb] # Generate unique IDs

    print(f"Knowledge Base has {len(documents_kb)} documents.\n")

    # --- Phase 1: Indexing with ChromaDB ---
    print("--- Initializing ChromaDB Client and Collection ---")
    # Create a persistent client (stores data on disk in a 'chroma_db' directory)
    # Or use: client = chromadb.Client() for an in-memory client (data lost on script exit)
    client = chromadb.PersistentClient(path="./chroma_db_store") # Data will be saved in this folder

    # Get or create the collection.
    # Chroma can automatically handle embedding generation if you provide an embedding function
    # compatible with its API. For simplicity with Gemini, we'll generate embeddings first.
    # We could also explore passing a genai.embed_content compatible function to Chroma later.
    
    # For now, we will generate embeddings ourselves and pass them to Chroma.
    # Note: If you run this script multiple times, collection.add() might error if IDs exist.
    # A real app would use get_or_create_collection and then potentially update/upsert logic.
    # For simplicity, let's try to get it, and if it fails (e.g. not found), create it.
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: '{collection_name}' with {collection.count()} items.")
        # If collection exists, we might want to skip adding documents if they are already there
        # For this example, if it exists and is not empty, we assume it's populated.
        # A more robust check would involve checking if specific IDs exist.
        if collection.count() == 0: # If collection exists but is empty
             print("Collection exists but is empty. Populating...")
             # Generate embeddings for our documents
             print("Generating document embeddings for Chroma...")
             document_embeddings_for_chroma = genai.embed_content(
                model=embedding_model_name,
                content=documents_kb,
                task_type="RETRIEVAL_DOCUMENT"
             )['embedding']
             collection.add(
                embeddings=document_embeddings_for_chroma,
                documents=documents_kb,
                metadatas=metadatas_kb,
                ids=ids_kb
             )
             print(f"Added {len(ids_kb)} documents to Chroma collection '{collection_name}'.")

    except Exception as e: # Catches if collection doesn't exist or other issues
        print(f"Collection '{collection_name}' not found or error: {e}. Creating and populating...")
        collection = client.create_collection(name=collection_name) # Add metadata for embedding function if Chroma handles it
        
        print("Generating document embeddings for Chroma...")
        document_embeddings_for_chroma = genai.embed_content(
            model=embedding_model_name,
            content=documents_kb,
            task_type="RETRIEVAL_DOCUMENT"
        )['embedding']

        collection.add(
            embeddings=document_embeddings_for_chroma, # We provide the embeddings
            documents=documents_kb,       # The text content
            metadatas=metadatas_kb,       # Associated metadata
            ids=ids_kb                    # Unique IDs for each document
        )
        print(f"Added {len(ids_kb)} documents to Chroma collection '{collection_name}'.")

    print(f"Chroma collection '{collection_name}' now has {collection.count()} items.\n")


    # --- Phase 2: Retrieval and Generation (For a User Query) ---
    queries_to_test = [
        "How tall is the Eiffel Tower?",
        "What currency is used in Japan?",
        "Who designed the Eiffel Tower?",
        "What is the capital of Germany?" # Not in our KB
    ]

    for user_query in queries_to_test:
        print(f"\n--- Processing User Query: \"{user_query}\" ---")

        # 1. Embed the User Query
        print("Embedding user query...")
        query_embedding = genai.embed_content(
            model=embedding_model_name,
            content=user_query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']

        # 2. Semantic Search with ChromaDB
        print("Querying ChromaDB...")
        top_k = 2
        # Query ChromaDB. We provide the query_embedding.
        # Chroma returns a dictionary with lists for 'ids', 'documents', 'metadatas', 'distances' (or 'similarities')
        results = collection.query(
            query_embeddings=[query_embedding], # Chroma expects a list of query embeddings
            n_results=top_k,
            # include=['documents', 'metadatas', 'distances'] # Specify what to return
        )

        retrieved_chroma_documents = results.get('documents', [[]])[0] # Get the list of document texts for the first query
        retrieved_chroma_metadatas = results.get('metadatas', [[]])[0]
        retrieved_chroma_distances = results.get('distances', [[]])[0] # Chroma often returns distances (lower is better)

        print("\n--- Retrieved Top-K Relevant Chunks from ChromaDB ---")
        if not retrieved_chroma_documents:
            print("No relevant documents found in ChromaDB.")
        for i, doc_text in enumerate(retrieved_chroma_documents):
            print(f"Chunk {i+1} (Distance: {retrieved_chroma_distances[i]:.4f}): \"{doc_text}\" (Metadata: {retrieved_chroma_metadatas[i]})")

        # 3. Augment Prompt (Stuff Context)
        if not retrieved_chroma_documents:
            context_for_llm = "No specific context found."
        else:
            context_for_llm = "\n".join(retrieved_chroma_documents)

        augmented_prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
If the answer is not found in the context or the context is 'No specific context found.', say "I don't have enough information from the provided documents to answer that."

Context:
{context_for_llm}

User Question: {user_query}

Answer:
"""
        print("\n--- Augmented Prompt for LLM ---")
        # print(augmented_prompt) # Keep it short for cleaner output for now

        # 4. Generate Response using the Generative LLM
        print("--- Generating Final Answer using LLM ---")
        generative_model = genai.GenerativeModel(generative_model_name)
        # Add safety settings to the generative model if desired
        # generative_model.safety_settings = ...
        final_response = generative_model.generate_content(augmented_prompt)

        print("\n--- Final Answer from LLM ---")
        print(final_response.text)
        print("-" * 50)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()