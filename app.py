
#CARD.AI APP

# ==== Section 0: LLM Setup ======


#Imports

import os
from PIL import Image
from streamlit_chat import message
from utils import *
import openai
from langchain import SQLDatabase
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, SQLDatabaseChain
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import numpy as np
from langchain.tools import PubmedQueryRun

from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import ConversationChain


#OPENAI setup -- needs secrets file
openai.api_key = OPENAI_API_KEY
# For Langchain
os.environ["OPENAI_API_KEY"] = openai.api_key

#For calculator function (just for testing)
llm_math_chain = LLMMathChain.from_llm(llm)

#Connect to vector database -  needs access to utils.py & have the index already setup with papers
qa = VectorDBQAWithSourcesChain.from_chain_type(
     llm=llm,
     chain_type="stuff",
     vectorstore=vectorstore
 )

#Initializing df for drug-genome interaction search
df = pd.read_csv("drug_genome_dgidb.csv", encoding= 'unicode_escape')
pd_agent = create_pandas_dataframe_agent(
    ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


#For Pubmed Search API Access
pubmedq = PubmedQueryRun()


#Tool list - edit to give new capabilities to choose from. See Langchain docs on tools
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="Pubmed Vector QA",
        func=qa,
        description="useful for when you need to answer questions about research in neurodegenerative diseases. State your sources in the final answer",
    ),
    Tool(
        name="Pubmed search",
        func=pubmedq,
        description="useful for when you need to answer questions about pubmed papers. Input should be in the form of a question containing full context",
    ),
    Tool(
        name="pandas agent",
        func=pd_agent.run,
        description="useful for when you need to answer questions about drug-genome interactions. State your sources in the final answer",
    ),

]


#Prompt setup - changing this could be helpful
chat_history = ""
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


#Initializes Agent with our prompt and tools
prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)


# ==== Section 1: Streamlit Settings ======

st.header("CARD.AI")

with st.sidebar:
    st.markdown("# Welcome to CARD.AI! 🧠🧬")
    st.markdown(
        "This demo allows you to search a curated database of Alzheimer's related literature in the style of **chatGPT**,\n "
        )
    st.markdown(
        "Our Chatbot levels up ChatGPT \n"
        "by using our curated database and granting it access to specialized tools. \n"
    )
    st.markdown("👩‍🏫 Original Developer: Wen Yang; adapted and modified by Chelsea Alvarado & Gracelyn Hill")
    st.markdown("---")
    st.markdown("# Under The Hood 🎩 🐇")
    st.markdown("How to Prevent Large Language Model (LLM) hallucination?")
    st.markdown("- **Pinecone**: vector database for Outside knowledge")
    st.markdown("- **Langchain**: to access tools and use chain-of-thought prompting")

# add CARD logo
card_img = Image.open('img/CARD-logo-white-print.png')
dti_img = Image.open('img/dti_img.jpeg')



st.sidebar.image(card_img)
st.sidebar.image(dti_img)



# ========== Section 3: Our Chatbot ============================



# Initialize agent
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)


# ========== Section 4. Display in chatbot style ===========


#Initializing session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'source' not in st.session_state:
    st.session_state['source'] = []


def clear_text():
    st.session_state["input"] = ""


# We will get the user's input by calling the get_text function
def get_text():
    
    input_text = st.text_input('Chat here! 💬', key="input")
    return input_text


user_input = get_text()

output_container = st.empty()

output_container = output_container.container()

if user_input:
    # source contain urls from Outside
    #chat history uses streamlit session state to send mesages to llm prompt
    chat_history = [val for pair in zip(st.session_state['generated'], st.session_state['past']) for val in pair]
    answer_container = output_container.chat_message("assistant", avatar="🧠")
    st_callback = StreamlitCallbackHandler(answer_container, expand_new_thoughts=False)

    output =  agent_chain.run({'input': user_input, 'chat_history':chat_history}, callbacks=[st_callback])
    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    
#Adding text to session state with chat bubbles
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i],  key=str(i))
        print(st.session_state["generated"][i])
        #message(st.session_state['source'][i],  key=str(i))
        message(st.session_state['past'][i], is_user=True,
                avatar_style="big-ears",  key=str(i) + '_user')


