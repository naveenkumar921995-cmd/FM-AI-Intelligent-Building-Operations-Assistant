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

h1, h2, h3 {
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🏢 FM AI System")

menu = st.sidebar.radio(
    "Navigation",
    ["AI Assistant", "Raise Ticket", "View Tickets"]
)

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
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

DB_PATH = "vectorstore/"

# ---------------- LOAD VECTORSTORE ---------------- #
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Vector DB not found. Creating...")

        from src.ingestion.ingest import load_documents, split_documents, create_vectorstore

        docs = load_documents()

        if not docs:
            st.error("❌ No documents found in data folder")
            st.stop()

        chunks = split_documents(docs)
        create_vectorstore(chunks)

    try:
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    except:
        st.warning("⚠️ Vector DB corrupted. Rebuilding...")

        from src.ingestion.ingest import load_documents, split_documents, create_vectorstore

        docs = load_documents()
        chunks = split_documents(docs)
        create_vectorstore(chunks)

        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# ✅ IMPORTANT (FIXED ISSUE)
vectorstore = load_vectorstore()

# ---------------- OEM DETECTION ---------------- #
def check_oem_required(text):
    keywords = [
        "compressor failure",
        "pcb fault",
        "controller fault",
        "motor winding",
        "vfd fault",
        "internal fault",
        "software issue"
    ]
    for k in keywords:
        if k in text.lower():
            return True
    return False

# ---------------- SMART RESPONSE ---------------- #
def generate_response(docs):
    combined = " ".join([doc.page_content for doc in docs])
    sentences = combined.split(".")

    insights = []
    for s in sentences:
        s = s.strip()
        if len(s) > 40:
            insights.append(s)
        if len(insights) >= 3:
            break

    return insights

# ---------------- AI ASSISTANT ---------------- #
if menu == "AI Assistant":

    st.title("🤖 FM AI Assistant")

    system_label = st.selectbox("Select System", list(system_map.keys()))
    system_filter = system_map[system_label]

    query = st.text_input("Ask your issue")

    if query:
        docs = vectorstore.similarity_search(query, k=5)
        docs = [d for d in docs if d.metadata.get("system") == system_filter]

        if not docs:
            st.warning("No relevant data found")
        else:
            st.subheader("🧠 Diagnosis")

            insights = generate_response(docs)
            oem_flag = False

            for i, point in enumerate(insights, 1):
                st.markdown(f"""
                <div class="card">
                    <b>{i}. {point}</b>
                </div>
                """, unsafe_allow_html=True)

                if check_oem_required(point):
                    oem_flag = True

            # ---------------- OEM ALERT ---------------- #
            if oem_flag:
                st.error("⚠️ OEM intervention required")

                if st.button("📨 Raise OEM Complaint"):
                    ticket_id = str(uuid.uuid4())[:8]

                    new_ticket = pd.DataFrame([{
                        "id": ticket_id,
                        "system": system_filter,
                        "issue": query,
                        "description": "OEM REQUIRED: " + " | ".join(insights),
                        "status": "Open",
                        "type": "OEM"
                    }])

                    if os.path.exists("tickets.csv"):
                        df = pd.read_csv("tickets.csv")
                        df = pd.concat([df, new_ticket], ignore_index=True)
                    else:
                        df = new_ticket

                    df.to_csv("tickets.csv", index=False)

                    st.success(f"✅ OEM Ticket Created: {ticket_id}")

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
            "status": "Open",
            "type": "General"
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

        ticket_id = st.text_input("Enter Ticket ID to Close")

        if st.button("Close Ticket"):
            df.loc[df["id"] == ticket_id, "status"] = "Closed"
            df.to_csv("tickets.csv", index=False)
            st.success("Ticket Closed")
    else:
        st.warning("No tickets found")
