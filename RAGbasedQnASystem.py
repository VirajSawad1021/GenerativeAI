import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import os
import chromadb
import uuid # For unique IDs

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash-latest' # Or 'gemini-pro' for text-only generation if preferred
CHROMA_PERSIST_PATH = "./scientist_db_store"
CHROMA_COLLECTION_NAME = "pioneering_scientists_collection"

# --- Document Data ---
DOCUMENTS_DATA = {
    "marie_curie": {
        "text": "Marie Skłodowska Curie (7 November 1867 – 4 July 1934) was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two different scientific fields. Her work was crucial in the development of X-rays in surgery. During World War I, Curie developed mobile radiography units to provide X-ray services to field hospitals. Despite her scientific successes, Curie faced significant gender and xenophobic discrimination from parts of the scientific community and the press. She died in 1934, aged 66, at a sanatorium in Sancellemoz, France, due to aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I.",
        "source": "Marie Curie Biography Snippet",
        "keywords_for_summary_tool": ["marie curie", "curie"]
    },
    "nikola_tesla": {
        "text": "Nikola Tesla (10 July 1856 – 7 January 1943) was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist best known for his contributions to the design of the modern alternating current (AC) electrical system. Born and raised in the Austrian Empire, Tesla studied engineering and physics in the 1870s without receiving a degree, gaining practical experience in the early 1880s working in telephony and at Continental Edison in the new electric power industry. He emigrated to the United States in 1884, where he would become a naturalized citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own. His alternating current induction motor and related polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and became the cornerstone of the polyphase system which that company would eventually market.",
        "source": "Nikola Tesla Biography Snippet",
        "keywords_for_summary_tool": ["nikola tesla", "tesla"]
    },
    "ada_lovelace": {
        "text": "Augusta Ada King, Countess of Lovelace (10 December 1815 – 27 November 1852), born Augusta Ada Byron, was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognize that the machine had applications beyond pure calculation, and published the first algorithm intended to be carried out by such a machine. As a result, she is often regarded as the first computer programmer. Lovelace's notes on the Analytical Engine include what is now recognized as the first published algorithm. She also speculated on the potential for computers to create graphics, compose music, and be used for both scientific and practical purposes, envisioning capabilities far beyond those imagined by most of her contemporaries.",
        "source": "Ada Lovelace Biography Snippet",
        "keywords_for_summary_tool": ["ada lovelace", "lovelace", "ada byron"]
    }
}

# --- Tool Definition (Python Function) ---
def get_document_summary(topic: str):
    """
    Provides a brief pre-defined summary for a known scientific topic or person mentioned in our documents.
    Use this if the user explicitly asks for a summary of Marie Curie, Nikola Tesla, or Ada Lovelace.

    Args:
        topic (str): The topic or person to summarize (e.g., "Marie Curie", "Nikola Tesla", "Ada Lovelace").

    Returns:
        dict: A dictionary containing the summary or an error message.
    """
    print(f"--- Python function get_document_summary(topic='{topic}') called ---")
    topic_lower = topic.lower()
    if any(keyword in topic_lower for keyword in DOCUMENTS_DATA["marie_curie"]["keywords_for_summary_tool"]):
        return {"summary": "Marie Curie was a pioneering physicist and chemist, the first woman to win a Nobel Prize, and the only person to win in two different scientific fields, known for her work on radioactivity."}
    elif any(keyword in topic_lower for keyword in DOCUMENTS_DATA["nikola_tesla"]["keywords_for_summary_tool"]):
        return {"summary": "Nikola Tesla was a Serbian-American inventor crucial to the development of the modern alternating current (AC) electrical system."}
    elif any(keyword in topic_lower for keyword in DOCUMENTS_DATA["ada_lovelace"]["keywords_for_summary_tool"]):
        return {"summary": "Ada Lovelace was an English mathematician considered the first computer programmer for her work on the Analytical Engine, recognizing its potential beyond calculation."}
    else:
        return {"summary": f"No pre-defined summary available for '{topic}'. I can only summarize Marie Curie, Nikola Tesla, or Ada Lovelace."}

# --- ChromaDB Setup and Indexing ---
def setup_chroma_collection():
    print("--- Initializing ChromaDB Client and Collection ---")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Retrieved existing collection: '{CHROMA_COLLECTION_NAME}' with {collection.count()} items.")
        if collection.count() == len(DOCUMENTS_DATA): # Basic check if already populated
            print("Collection appears to be already populated.")
            return collection
        elif collection.count() > 0: # If partially populated, best to recreate for this example
            print(f"Collection exists but has unexpected item count ({collection.count()}). Recreating for consistency in this example.")
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
            raise chromadb.errors.CollectionNotFoundError("Recreating collection") # Force recreation
            
    except Exception: # Handles CollectionNotFoundError and others during get
        print(f"Creating and populating new collection: '{CHROMA_COLLECTION_NAME}'...")
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

        docs_to_embed = []
        metadatas_to_store = []
        ids_to_store = []

        for key, data in DOCUMENTS_DATA.items():
            docs_to_embed.append(data["text"])
            metadatas_to_store.append({"scientist_name": key.replace("_", " ").title(), "source": data["source"]})
            ids_to_store.append(key) # Use scientist key as ID

        print("Generating document embeddings for Chroma...")
        document_embeddings = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=docs_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )['embedding']

        collection.add(
            embeddings=document_embeddings,
            documents=docs_to_embed,
            metadatas=metadatas_to_store,
            ids=ids_to_store
        )
        print(f"Added {len(ids_to_store)} documents to Chroma collection '{CHROMA_COLLECTION_NAME}'.")
    
    print(f"Chroma collection '{CHROMA_COLLECTION_NAME}' ready with {collection.count()} items.\n")
    return collection

# --- Main Q&A Bot Logic ---
def run_qna_bot():
    chroma_collection = setup_chroma_collection()

    safety_settings_config = [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}, # Stricter now
    ]

    generative_model_instance = genai.GenerativeModel(
        GENERATIVE_MODEL_NAME,
        safety_settings=safety_settings_config,
        tools=[get_document_summary] # Provide the function for automatic calling
    )

    chat_session = generative_model_instance.start_chat(enable_automatic_function_calling=True)

    print("\n--- Responsible Document Q&A Bot ---")
    print("Ask me questions about Marie Curie, Nikola Tesla, or Ada Lovelace.")
    print("You can also ask for a 'summary of [scientist name]'. Type 'quit' to end.")
    print("-" * 50)

    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "quit", "exit", "see ya"]

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in farewells:
            print("Bot: Goodbye! Have a great day.")
            break

        if user_input.lower() in greetings:
            print("Bot: Hello! How can I help you with information about our featured scientists?")
            print("-" * 50)
            continue
        
        print("Bot: Thinking...")
        
        # --- Simplified Router: Check if user explicitly asks for a summary ---
        # The LLM itself, with the tool description, should also pick this up.
        # This explicit check is just to make it more direct for this project's scope.
        # A more advanced router would use another LLM call.
        
        # Let the LLM decide if the summary tool is appropriate based on its description and user input.
        # If not, it will try to answer directly. We will then perform RAG if needed,
        # but for this combined automatic example, let's see how the LLM handles it
        # when RAG context is directly added to its *next* turn if it doesn't use a tool or answer satisfyingly.
        # For now, let's just send the user input and let auto function calling try.
        # If it doesn't use the function, its response will be based on its general knowledge.

        # Send user input, LLM might use a tool OR answer from general knowledge
        try:
            llm_response = chat_session.send_message(user_input)
            
            # If the LLM didn't use the function call (indicated by no specific print from our function)
            # and the answer seems generic or "I don't know", we can then try RAG.
            # For this example, we'll make the RAG step more explicit if no tool was used.
            # A more sophisticated agent would have a loop here.

            # We need to inspect the llm_response to see if a function was called.
            # The chat_session.history will show the function call and response parts if it happened.
            called_function_this_turn = False
            if len(chat_session.history) > 1: # Need at least user input and model response
                last_model_turn = chat_session.history[-1] # The model's most recent complete turn
                if last_model_turn.role == 'model':
                    for part in last_model_turn.parts:
                        if hasattr(part, 'function_call') or hasattr(part, 'function_response'): # Check previous parts in history
                            # A bit tricky to check if *this current interaction* involved an auto-call
                            # without seeing the intermediate steps which are hidden by auto-mode.
                            # We rely on our print statement inside get_document_summary for now.
                            # A better check would be if the response text *is* the summary.
                            # For simplicity, if the response is short and "No pre-defined summary...", we know tool was tried.
                            if "No pre-defined summary available for" in llm_response.text or \
                               any(summary_text in llm_response.text for summary_text in [
                                   "Marie Curie was a pioneering", "Nikola Tesla was a Serbian-American", "Ada Lovelace was an English mathematician"
                               ]):
                                called_function_this_turn = True
            
            # If a function was called and provided a summary, we print its response.
            # If not, or if the user asked a general question, we proceed to RAG.
            if called_function_this_turn:
                print(f"Bot: {llm_response.text}") # This is the LLM's response after using the tool
            else:
                # --- Perform RAG if no function was called or if LLM didn't answer well ---
                print("Bot: (Didn't use summary tool, attempting RAG...)")
                query_embedding = genai.embed_content(
                    model=EMBEDDING_MODEL_NAME,
                    content=user_input,
                    task_type="RETRIEVAL_QUERY"
                )['embedding']

                rag_results = chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2 # Get top 2
                )
                retrieved_docs = rag_results.get('documents', [[]])[0]

                if not retrieved_docs:
                    context_for_llm = "No specific context found in documents."
                else:
                    context_for_llm = "\n\n".join(retrieved_docs)

                # Now, send a new message to the chat session WITH the RAG context.
                # The LLM will use this context. It still has access to tools if relevant.
                rag_augmented_input = f"""Please answer the following user question based ONLY on the provided context.
If the answer is not in the context, state that you don't have enough information from the documents.
You can still use your 'get_document_summary' tool if the user explicitly asks for a summary of a known person, even with this context.

Context from Documents:
{context_for_llm}

Original User Question: {user_input}

Answer:"""
                print("Bot: Thinking with RAG context...")
                final_rag_response = chat_session.send_message(rag_augmented_input)
                print(f"Bot: {final_rag_response.text}")

        except Exception as e:
            print(f"Bot: I encountered an issue: {e}")
            # Potentially log the error or provide a more user-friendly message

        # Safety Feedback (Optional: can be verbose for chat)
        # if llm_response.prompt_feedback: print(f"DEBUG: Prompt Feedback: {llm_response.prompt_feedback}")
        # if llm_response.candidates and llm_response.candidates[0].safety_ratings: print(f"DEBUG: Candidate Safety: {llm_response.candidates[0].safety_ratings}")
        print("-" * 50)

if __name__ == "__main__":
    run_qna_bot()