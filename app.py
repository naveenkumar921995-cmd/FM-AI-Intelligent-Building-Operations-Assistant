import streamlit as st
import os
import pandas as pd
import uuid

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="FM AI Assistant", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
.main { background-color: #f7f9fc; color: #1f2937; }
.card {
    padding: 15px;
    border-radius: 8px;
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🏢 FM AI System")

menu = st.sidebar.radio("Navigation", ["AI Assistant", "Raise Ticket", "View Tickets"])

# ---------------- SYSTEM MAP ---------------- #
system_map = {
    "HVAC": "hvac",
    "Fire": "fire",
    "Electrical": "electrical",
    "DG": "dg",
    "STP": "stp",
    "WTP": "wtp",
    "Lift": "lift",
    "CCTV": "cctv",
    "Access Control": "access_control",
    "Soft Services": "soft_services"
}

# ---------------- EMBEDDINGS ---------------- #
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_PATH = "vectorstore/"

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# ---------------- AI ASSISTANT ---------------- #
if menu == "AI Assistant":

    st.title("🤖 FM AI Assistant")

    system_label = st.selectbox("Select System", list(system_map.keys()))
    system_filter = system_map[system_label]

    query = st.text_input("Ask your issue")

    if query:
        docs = vectorstore.similarity_search(query, k=5)
        docs = [d for d in docs if d.metadata.get("system") == system_filter]

        st.subheader("Diagnosis")

        for i, doc in enumerate(docs[:3], 1):
            st.markdown(f"""
            <div class="card"><b>{i}. {doc.page_content}</b></div>
            """, unsafe_allow_html=True)

# ---------------- RAISE TICKET ---------------- #
elif menu == "Raise Ticket":

    st.title("📝 Raise Complaint")

    system = st.selectbox("System", list(system_map.keys()))
    issue = st.text_input("Issue Title")
    description = st.text_area("Detailed Description")

    if st.button("Submit Ticket"):

        ticket_id = str(uuid.uuid4())[:8]

        new_ticket = pd.DataFrame([{
            "id": ticket_id,
            "system": system,
            "issue": issue,
            "description": description,
            "status": "Open"
        }])

        if os.path.exists("tickets.csv"):
            df = pd.read_csv("tickets.csv")
            df = pd.concat([df, new_ticket], ignore_index=True)
        else:
            df = new_ticket

        df.to_csv("tickets.csv", index=False)

        st.success(f"✅ Ticket Created! ID: {ticket_id}")

# ---------------- VIEW TICKETS ---------------- #
elif menu == "View Tickets":

    st.title("📊 Ticket Dashboard")

    if os.path.exists("tickets.csv"):
        df = pd.read_csv("tickets.csv")

        st.dataframe(df)

        # Update status
        ticket_id = st.text_input("Enter Ticket ID to Close")

        if st.button("Close Ticket"):
            df.loc[df["id"] == ticket_id, "status"] = "Closed"
            df.to_csv("tickets.csv", index=False)
            st.success("Ticket Closed")

    else:
        st.warning("No tickets found")
