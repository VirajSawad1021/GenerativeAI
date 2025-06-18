import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np # For cosine similarity later
from sklearn.metrics.pairwise import cosine_similarity # A common way to calculate it

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

    # --- Specify the Embedding Model ---
    # Using text-embedding-004 as it's a generally available and good model.
    # Other models like 'text-embedding-gecko-001' or newer ones might also be available.
    # The new 'models/embedding-001' (if you're using the direct genai.embed_content) is another option,
    # or 'text-embedding-004' if that's the preferred alias.
    # Let's align with a commonly cited model that supports task types.
    # The documentation refers to 'text-embedding-004' and also newer models.
    # If `text-embedding-004` is not found, try `models/text-embedding-004` or `embedding-001` (newer name for default model)
    # The search results indicate `text-embedding-004` is a valid model name.
    embedding_model_name = "text-embedding-004"


    print(f"--- Using Embedding Model: {embedding_model_name} ---")

    # --- 1. Embedding a Single Piece of Text (e.g., for a search query) ---
    query_text = "What is the best way to learn about artificial intelligence?"
    print(f"\nText to embed (as a query): \"{query_text}\"")

    # When embedding a query for retrieval, use task_type="RETRIEVAL_QUERY"
    # When embedding documents to be retrieved, use task_type="RETRIEVAL_DOCUMENT"
    query_embedding_response = genai.embed_content(
        model=embedding_model_name,
        content=query_text,
        task_type="RETRIEVAL_QUERY", # Crucial for RAG
        # title="Optional title for the query" # Can sometimes help if applicable
    )

    query_embedding_vector = query_embedding_response['embedding']
    print(f"Embedding vector (first 5 dimensions): {query_embedding_vector[:5]}")
    print(f"Embedding vector dimension: {len(query_embedding_vector)}")


    # --- 2. Embedding a Batch of Documents (e.g., for indexing) ---
    documents_to_embed = [
        "Artificial intelligence research has advanced significantly in recent years.",
        "Machine learning is a subfield of AI focused on algorithms that learn from data.",
        "Deep learning, a type of machine learning, uses neural networks with many layers.",
        "The best way to make tea is by first boiling the water." # A semantically different sentence
    ]
    print(f"\nDocuments to embed (as documents for retrieval): {documents_to_embed}")

    # For batch embedding, 'content' is a list of strings.
    # Use task_type="RETRIEVAL_DOCUMENT"
    # Note: Some models have limits on batch size. text-embedding-004 supports batching.
    # If the chosen model doesn't support batching for the 'content' list directly,
    # you might need to loop and call embed_content for each document.
    # However, text-embedding-004 is generally good with batching content.
    
    # The API expects a list of contents for batching.
    document_embeddings_response = genai.embed_content(
        model=embedding_model_name,
        content=documents_to_embed, # Pass the list directly
        task_type="RETRIEVAL_DOCUMENT"
    )

    document_embedding_vectors = document_embeddings_response['embedding'] # This will be a list of lists (embeddings)

    print("\n--- Document Embeddings ---")
    for i, doc_text in enumerate(documents_to_embed):
        print(f"Document {i+1}: \"{doc_text[:50]}...\"")
        print(f"  Embedding (first 5 dimensions): {document_embedding_vectors[i][:5]}")
        print(f"  Embedding vector dimension: {len(document_embedding_vectors[i])}")


    # --- 3. Calculating Semantic Similarity (Cosine Similarity) ---
    # We'll compare the query embedding with each document embedding.
    print("\n--- Semantic Similarity (Cosine Similarity) between Query and Documents ---")
    
    # Reshape query_embedding_vector to be a 2D array for cosine_similarity function
    query_vec_2d = np.array(query_embedding_vector).reshape(1, -1)

    for i, doc_vec in enumerate(document_embedding_vectors):
        doc_vec_2d = np.array(doc_vec).reshape(1, -1)
        similarity = cosine_similarity(query_vec_2d, doc_vec_2d)[0][0] # Get the single similarity score
        print(f"Similarity between Query and Document {i+1} ('{documents_to_embed[i][:30]}...'): {similarity:.4f}")


except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting tips:")
    print("- Ensure your API key is correctly configured and has permissions for the embedding model.")
    print(f"- Double-check the embedding model name: '{embedding_model_name}'. Try 'models/embedding-001' or 'embedding-001' if '{embedding_model_name}' fails.")
    print("- Check quotas for the embedding API in your Google Cloud project if applicable.")