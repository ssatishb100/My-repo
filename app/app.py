import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
import os

st.set_page_config(page_title="AI Ticket Auditor", layout="wide")
st.title("🤖 AI Ticket Auditor (History-Based)")

# Securely get API Key from Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- SIDEBAR: TRAINING ---
with st.sidebar:
    st.header("1. Training Data")
    uploaded_file = st.file_uploader("Upload Jira CSV", type="csv")
    
    if uploaded_file and st.button("Train from History"):
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        documents = []
        # Logic to find relevant review columns
        comment_cols = [c for c in df.columns if 'Comment' in c]
        
        for i, row in df.iterrows():
            desc = str(row.get('Description', ''))
            all_comments = " ".join([str(row[c]) for c in comment_cols if pd.notna(row[c])])
            key = str(row.get('Issue key', f'ID-{i}'))
            
            if any(word in all_comments.lower() for word in ["reviewed", "approved", "fix"]):
                doc = Document(page_content=f"OLD TICKET: {desc}\nREVIEW: {all_comments}", 
                               metadata={"source": key})
                documents.append(doc)
        
        st.session_state.vector_db = Chroma.from_documents(documents, OpenAIEmbeddings(api_key=api_key))
        st.success(f"Trained on {len(documents)} patterns!")

# --- MAIN AREA: AUDITING ---
st.header("2. Run New Review")
new_ticket = st.text_area("Paste the ticket description you want to audit:", height=200)

if st.button("Analyze with AI"):
    if st.session_state.vector_db:
        # Search & Prompt logic
        docs = st.session_state.vector_db.similarity_search(new_ticket, k=2)
        context = "\n---\n".join([d.page_content for d in docs])
        sources = [d.metadata['source'] for d in docs]
        
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
        response = llm.invoke(f"Use these past reviews:\n{context}\n\nReview this new ticket:\n{new_ticket}")
        
        st.subheader("AI Feedback")
        st.write(response.content)
        st.info(f"Historical Sources: {', '.join(sources)}")
    else:
        st.error("Please upload and train on a CSV first.")
