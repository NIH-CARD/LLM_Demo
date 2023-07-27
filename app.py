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

pubmedq = PubmedQueryRun()
st.header("CARD.AI")
# Tools setup
llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY, streaming=True)
llm_math_chain = LLMMathChain.from_llm(llm)
qa = VectorDBQAWithSourcesChain.from_chain_type(
     llm=llm,
     chain_type="stuff",
     vectorstore=vectorstore
 )

df = pd.read_csv("drug_genome_dgidb.csv", encoding= 'unicode_escape')
pd_agent = create_pandas_dataframe_agent(
    ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

chat_history = ""
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
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)



# class DirtyState:
#     NOT_DIRTY = "NOT_DIRTY"
#     DIRTY = "DIRTY"
#     UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


# def get_dirty_state() -> str:
#     return st.session_state.get("dirty_state", DirtyState.NOT_DIRTY)


# def set_dirty_state(state: str) -> None:
#     st.session_state["dirty_state"] = state


# def with_clear_container(submit_clicked: bool) -> bool:
#     if get_dirty_state() == DirtyState.DIRTY:
#         if submit_clicked:
#             set_dirty_state(DirtyState.UNHANDLED_SUBMIT)
#             st.experimental_rerun()
#         else:
#             set_dirty_state(DirtyState.NOT_DIRTY)

#     if submit_clicked or get_dirty_state() == DirtyState.UNHANDLED_SUBMIT:
#         set_dirty_state(DirtyState.DIRTY)
#         return True

#     return False

openai.api_key = OPENAI_API_KEY
# For Langchain
os.environ["OPENAI_API_KEY"] = openai.api_key


# ==== Section 1: Streamlit Settings ======
with st.sidebar:
    st.markdown("# Welcome to our LLM Chatbot 🙌")
    st.markdown(
        "This demo allows you to search a curated database of Alzheimer's related literature in the style of **chatGPT** \n"
        )
    st.markdown(
        "Unlike chatGPT, our chatbot can't make stuff up\n"
        "since it uses our curated database. \n"
    )
    st.markdown("👩‍🏫 Original Developer: Wen Yang; adapted and modified by Chelsea Alvarado")
    st.markdown("---")
    st.markdown("# Under The Hood 🎩 🐇")
    st.markdown("How to Prevent Large Language Model (LLM) hallucination?")
    st.markdown("- **Pinecone**: vector database for Outside knowledge")
    st.markdown("- **Langchain**: to remember the context of the conversation")

# Homepage title
st.title("Demo")
# Hero Image
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
        description="""write a python script using pandas to access OmicSynth Data about the druggability of genes for AD, ALS, FTLD, LBD. PD and PSP. You have access to the following columns:
gene_name	gene_claim_name	entrez_id	interaction_claim_source	interaction_types	drug_claim_name	drug_claim_primary_name	drug_name	drug_concept_id	interaction_group_score	PMIDs	ensembl_gene_id	druggability_tier	hgnc_names	chr_b37	start_b37	end_b37	strand	description	no_of_gwas_regions	small_mol_druggable	bio_druggable	adme_gene
 These columns include snRNA-seq expression profiles for genes, Genes expressed in different cell types for diseases, drug interactions for genes.
""",
    ),
]
output_container = st.empty()

output_container = output_container.container()

if user_input:
    # source contain urls from Outside
    chat_history = [val for pair in zip(st.session_state['generated'], st.session_state['past']) for val in pair]
    answer_container = output_container.chat_message("assistant", avatar="🧠")
    st_callback = StreamlitCallbackHandler(answer_container, expand_new_thoughts=False)

    output =  agent_chain.run({'input': user_input, 'chat_history':chat_history}, callbacks=[st_callback])
    #answer_container.write(output)
    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    # Display source urls
    

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i],  key=str(i))
        print(st.session_state["generated"][i])
        #message(st.session_state['source'][i],  key=str(i))
        message(st.session_state['past'][i], is_user=True,
                avatar_style="big-ears",  key=str(i) + '_user')


# with st.form(key="form"):
#     user_input = st.text_input("Or, ask your neuro questions!")
#     submit_clicked = st.form_submit_button("Go!")

# output_container = st.empty()
# output_container = output_container.container()
# output_container.chat_message("user").write(user_input)

# answer_container = output_container.chat_message("assistant", avatar="🧠")
# st_callback = StreamlitCallbackHandler(answer_container)

# answer = mrkl.run(user_input, callbacks=[st_callback])

# answer_container.write(answer)

