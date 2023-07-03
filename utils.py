import pinecone
import streamlit as st
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ------OpenAI: LLM---------------
OPENAI_API_KEY = st.secrets["openai_api_key"]
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo-16k',
    temperature=0.0
)

# ------OpenAI: Embed model-------------
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

# --- Pinecone ------
pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone_environment = st.secrets["pinecone_environ"]
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = 'os-chatgpt'
index = pinecone.Index(index_name)
text_field = "text"
vectorstore = Pinecone(index, embed.embed_query, text_field)


#  ======= Langchain ChatDBQA with source chain =======
def qa_with_sources(query):
    qa = VectorDBQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        vectorstore=vectorstore
    )

    response = qa(query)
    return response
