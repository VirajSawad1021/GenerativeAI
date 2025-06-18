import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)

    # --- 0. Configuration ---
    embedding_model_name = "text-embedding-004" # Or 'models/embedding-001'
    generative_model_name = 'gemini-1.5-flash-latest' # For generating the final answer

    # --- 1. Our "Knowledge Base" (simple list of documents/chunks) ---
    documents = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
        "Constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design.",
        "The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and is the tallest structure in Paris.",
        "Millions of people ascend it every year, making it one of the most visited paid monuments in the world.",
        "The official currency of Japan is the Yen.",
        "Japan is an island country in East Asia, located in the northwest Pacific Ocean."
    ]
    print(f"Knowledge Base has {len(documents)} documents.\n")

    # --- Phase 1: Indexing (Generate and "Store" Embeddings for our Documents) ---
    # In a real app, this is done once and stored in a vector DB.
    print("--- Indexing Documents (Generating Embeddings) ---")
    document_embeddings = genai.embed_content(
        model=embedding_model_name,
        content=documents,
        task_type="RETRIEVAL_DOCUMENT"
    )['embedding'] # Extract the list of embedding vectors
    print(f"Generated {len(document_embeddings)} document embeddings, each with dimension {len(document_embeddings[0]) if document_embeddings else 'N/A'}.\n")


    # --- Phase 2: Retrieval and Generation (For a User Query) ---
    # user_query = "How tall is the Eiffel Tower?"
    user_query = "What is my name?"
    # user_query = "Who designed the Eiffel Tower?"

    print(f"--- Processing User Query: \"{user_query}\" ---")

    # 1. Embed the User Query
    print("Embedding user query...")
    query_embedding = genai.embed_content(
        model=embedding_model_name,
        content=user_query,
        task_type="RETRIEVAL_QUERY"
    )['embedding'] # Extract the single query embedding vector

    # 2. Semantic Search (Find relevant documents from our "Knowledge Base")
    # We'll calculate cosine similarity between the query and all document embeddings.
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        # Reshape for sklearn's cosine_similarity function
        similarity = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(doc_embedding).reshape(1, -1)
        )[0][0]
        similarities.append((similarity, i, documents[i])) # Store similarity, index, and text

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)

    # 3. Select Top-K Relevant Documents for Context
    top_k = 2 # Let's take the top 2 most relevant documents
    retrieved_chunks = [doc_text for similarity, index, doc_text in similarities[:top_k]]

    print("\n--- Retrieved Top-K Relevant Chunks ---")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i+1} (Similarity: {similarities[i][0]:.4f}): \"{chunk}\"")

    # 4. Augment Prompt (Stuff Context)
    context_for_llm = "\n".join(retrieved_chunks)
    augmented_prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
If the answer is not found in the context, say "I don't have enough information from the provided documents to answer that."

Context:
{context_for_llm}

User Question: {user_query}

Answer:
"""
    print("\n--- Augmented Prompt for LLM ---")
    print(augmented_prompt)

    # 5. Generate Response using the Generative LLM
    print("--- Generating Final Answer using LLM ---")
    generative_model = genai.GenerativeModel(generative_model_name)
    final_response = generative_model.generate_content(augmented_prompt)

    print("\n--- Final Answer from LLM ---")
    print(final_response.text)

except Exception as e:
    print(f"An error occurred: {e}")