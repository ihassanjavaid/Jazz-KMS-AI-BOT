import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Azure OpenAI Config ----
AZURE_OPENAI_API_KEY = os.getenv("gpt_4_o_mini_AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("gpt_4_o_mini_AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("gpt_4_o_mini_AZURE_OPENAI_DEPLOYMENT_NAME")   # your deployment name in Azure
API_VERSION = os.getenv("gpt_4_o_mini_AZURE_OPENAI_API_VERSION")

st.caption("Jazz\'s very own smart assistant — powered intelligently by KMS.")

st.markdown(
    """
    <style>
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image("jazz_logo.png", width=60)
    st.markdown("### Jazz AI KMS-Bot")
    st.markdown("---")
    st.write("CX\'s very own smart agent to answer your queries in the blink of an eye, leveraging the knowledge of KMS.")
    st.markdown("---")
    st.markdown("**CX – D & U**")

with st.expander("How does it work?"):
    st.write("""
    Step 1: Upload a PDF file **(of a KMS page)**.  
    Step 2. Ask a question aobut it in the chat box.  
    Step 3. The AI reads the KMS\'s PDF and returns an answer.  
    """)


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Jazz AI Agent", 
    page_icon="jazz_logo.png",   # You can use emoji OR path to an image (e.g., "logo.png")
    layout="centered"
)

# ---------- COMPANY LOGO ----------
# st.image("jazz_logo.png", width=80)  
st.title("Jazz AI KMS-Bot")# Replace with your logo file path

# Upload PDF
pdf_file = st.file_uploader("Please upload a KMS page\'s PDF:", type=["pdf"])

# Keep PDF text in session state (so it persists across chat turns)
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if pdf_file and not st.session_state.pdf_text:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    st.session_state.pdf_text = "\n\n".join([chunk.page_content for chunk in chunks])

# Show chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# st.markdown(
#     """
#     <hr style="margin-top:50px; margin-bottom:10px;">
#     <div style="text-align: center; color: gray;">
#         Made by <b>CX – Design & Usability</b>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            # background: #f9f9f9;   /* light background so it blends */
            padding: 8px;
            text-align: center;
            font-size: 14px;
            color: gray;
            # border-top: 1px solid #e0e0e0;
            z-index: 100;
        }
    </style>
    <div class="footer">
        Made by: <b>CX – Design & Usability</b>
    </div>
    """,
    unsafe_allow_html=True
)

# Chat input (question box)
if prompt := st.chat_input("Chat with Jazz AI KMS-Bot..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LLM setup
    llm = AzureChatOpenAI(
        azure_deployment=DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0.2,
        max_tokens=500
    )

    # Prompt
    template = """
    You are an assistant. Your name is \"Jazz AI Agent KMS BOT\". The following is the content of a PDF:

    {context}

    Based on this content, answer the question:
    {question}
    """
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate answer
    with st.chat_message("Jazz AI"):
        with st.spinner("Thinking..."):
            answer = chain.run(context=st.session_state.pdf_text, question=prompt)
            st.markdown(answer)

    # Save answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
