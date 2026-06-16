"""
app.py — Chef Marco chatbot with Hybrid RAG
  1. Embed the user query
  2. Search vector_store.json with cosine similarity
  3. If best score >= SCORE_THRESHOLD  →  answer from document chunk
  4. Else                               →  search Wikipedia, clean with regex, answer from snippets
  5. Tag every reply with its source

Dependencies (pip install):
    streamlit google-generativeai wikipedia-api
"""

import json
import math
import re
import streamlit as st
import google.generativeai as genai
import wikipediaapi

VECTOR_STORE_FILE = "vector_store.json"
EMBED_MODEL       = "models/gemini-embedding-001"
GEN_MODEL         = "gemini-3.5-flash"
TOP_K             = 4       # chunks to retrieve
SCORE_THRESHOLD   = 0.35    


with open("geminiapikey22.txt", "r") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)
model = genai.GenerativeModel(GEN_MODEL)

wiki = wikipediaapi.Wikipedia(
    user_agent="ChefMarcoBot/1.0",
    language="en",
)


@st.cache_resource
def load_vector_store():
    with open(VECTOR_STORE_FILE, "r") as f:
        return json.load(f)

vector_store = load_vector_store()



def cosine_similarity(a, b):
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def embed(text, task="retrieval_query"):
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type=task,
    )
    return result["embedding"]


def semantic_search(query_vec, top_k=TOP_K):
    """Return top-K chunks sorted by cosine similarity."""
    scored = sorted(
        vector_store,
        key=lambda item: cosine_similarity(query_vec, item["embedding"]),
        reverse=True,
    )
    return scored[:top_k]


def clean_wiki_text(raw: str) -> str:
    """
    Use regex to strip Wikipedia boilerplate:
    - citation markers like [1], [citation needed]
    - section headers (== Header ==)
    - edit markers
    - excessive whitespace
    """
    text = re.sub(r"\[[\w\s]+\]",  "",   raw)   # [1], [citation needed]
    text = re.sub(r"={2,}.*?={2,}", "",  text)   # == Section ==
    text = re.sub(r"\(listen\)",    "",   text)   # (listen) audio markers
    text = re.sub(r"\n{3,}",        "\n\n", text) # collapse blank lines
    text = re.sub(r"[ \t]{2,}",     " ",   text)  # collapse spaces
    return text.strip()


def wikipedia_search(query: str, max_chars: int = 2000) -> str:
    """
    Search Wikipedia for the query.
    Returns cleaned summary text or an error string.
    """
    page = wiki.page(query)
    if not page.exists():
        # Try a shorter version of the query (first 3 words)
        shorter = " ".join(query.split()[:3])
        page = wiki.page(shorter)

    if not page.exists():
        return ""

    raw = page.summary
    cleaned = clean_wiki_text(raw)
    return cleaned[:max_chars]



BASE_PROMPT = """
You are Chef Marco, a master chef of personal growth and human potential.
You speak with warmth, wisdom, and tough love ,like a mentor who truly believes in people.
You remember everything shared in this conversation and personalize your advice accordingly.

## Your Philosophy:
- Discipline is the foundation of all growth. Motivation is a guest; discipline is the host.
- Small daily habits compound into massive results over time.
- Failure is feedback, not defeat. Every setback is data.
- Progress over perfection  always.

## Your Style:
- Use cooking metaphors naturally when they fit.
- Ask follow-up questions to personalize advice ,never give generic answers.
- Be direct and honest, but always encouraging.
- Keep responses concise and actionable.
- Recommend books or thinkers from the context when relevant.

## Source context for this query:
Source: {source_label}

{context}

## Rules:
- Answer using ONLY the context above.
- If context is insufficient, say so honestly and answer from general wisdom.
- At the end of your reply, add a one-line footer: "Source: {source_label}"
- Never break character. You are always Chef Marco.
"""


st.set_page_config(page_title="Chef Marco", page_icon="👨‍🍳")
st.title("👨‍🍳 Chef Marco")
st.write("Your guide to growth, discipline, and personal development.")
st.divider()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.header("RAG Debug")
    show_debug  = st.toggle("Show retrieval details", value=False)
    score_thresh = st.slider(
        "Score threshold",
        min_value=0.10, max_value=0.70,
        value=SCORE_THRESHOLD, step=0.05,
        help="Above → use document. Below → search Wikipedia."
    )
    st.caption(f"Vector store: {len(vector_store)} chunks | Top-K: {TOP_K}")


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Ask Chef Marco anything...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.chat_history.append({
        "role": "user", "content": user_input
    })

    
    query_vec = embed(user_input, task="retrieval_query")

    
    top_chunks  = semantic_search(query_vec)
    best_score  = cosine_similarity(query_vec, top_chunks[0]["embedding"])
    best_chunk  = top_chunks[0]["text"]

    
    if best_score >= score_thresh:
        # Document path
        context      = "\n\n---\n\n".join(c["text"] for c in top_chunks)
        source_label = "knowledge_base.txt"
        source_path  = "document"
    else:
        # Web path: Wikipedia
        wiki_text    = wikipedia_search(user_input)
        if wiki_text:
            context      = wiki_text
            source_label = "Wikipedia"
            source_path  = "wikipedia"
        else:
            context      = "No relevant information found in the document or Wikipedia."
            source_label = "general knowledge"
            source_path  = "none"

    
    if show_debug:
        with st.sidebar:
            st.subheader("This query")
            st.metric("Best score", f"{best_score:.3f}")
            st.metric("Threshold",  f"{score_thresh:.2f}")
            st.metric("Path used",  source_path)
            st.divider()
            if source_path == "document":
                for i, c in enumerate(top_chunks, 1):
                    score = cosine_similarity(query_vec, c["embedding"])
                    st.markdown(f"**Chunk {i}** — `{score:.3f}`")
                    st.caption(c["text"][:250] + "...")
                    st.divider()
            else:
                st.caption(context[:400] + "...")

    
    conversation_text = ""
    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "Chef Marco"
        conversation_text += f"{role}: {msg['content']}\n"

    system_prompt = BASE_PROMPT.format(
        source_label=source_label,
        context=context,
    )

    full_prompt = f"""
{system_prompt}

## Conversation so far:
{conversation_text}

Chef Marco:
"""

    
    try:
        with st.chat_message("assistant"):
            with st.spinner("Chef Marco is thinking..."):
                response        = model.generate_content(full_prompt)
                assistant_reply = response.text
                st.write(assistant_reply)
    except Exception as e:
        assistant_reply = f"Something went wrong in the kitchen: {str(e)}"
        with st.chat_message("assistant"):
            st.error(assistant_reply)

    st.session_state.chat_history.append({
        "role": "assistant", "content": assistant_reply
    })

st.divider()
if st.button("Clear chat"):
    st.session_state.chat_history = []
    st.rerun()
