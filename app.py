
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
st.header("CARD.AI")


#OPENAI setup -- needs secrets file
openai.api_key = OPENAI_API_KEY
# For Langchain
os.environ["OPENAI_API_KEY"] = openai.api_key
# We will get the user's input by calling the get_text function
def get_text():
    
    input_text = st.text_input('Chat here! 💬', key="input")
    return input_text


user_input = get_text()


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
from langchain.chains import APIChain
hugeamp = APIChain.from_llm_and_api_docs(llm, """{"openapi":"3.0.2","info":{"title":"BioIndex","version":"0.1.0"},"paths":{"/api/bio/indexes":{"get":{"tags":["bio"],"summary":"Api List Indexes","description":"Return all queryable indexes. This also refreshes the internal\ncache of the table so the server doesn't need to be bounced when\nthe table is updated (very rare!).","operationId":"api_list_indexes_api_bio_indexes_get","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}},"/api/bio/match/{index}":{"get":{"tags":["bio"],"summary":"Api Match","description":"Return all the unique keys for a value-indexed table.","operationId":"api_match_api_bio_match__index__get","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"},{"required":true,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"},{"required":false,"schema":{"title":"Limit","type":"integer"},"name":"limit","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/bio/count/{index}":{"get":{"tags":["bio"],"summary":"Api Count Index","description":"Query the database and estimate how many records will be returned.","operationId":"api_count_index_api_bio_count__index__get","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"},{"required":false,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/bio/all/{index}":{"get":{"tags":["bio"],"summary":"Api All","description":"Query the database and return ALL records for a given index. If the\ntotal number of bytes read exceeds a pre-configured server limit, then\na 413 response will be returned. If multiple indexes share a name\nwith different arity it'll throw a 400.","operationId":"api_all_api_bio_all__index__get","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"},{"required":false,"schema":{"title":"Fmt","type":"string","default":"row"},"name":"fmt","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}},"head":{"tags":["bio"],"summary":"Api Test All","description":"Query the database fetch ALL records for a given index. Don't read\nthe records from S3, but instead set the Content-Length to the total\nnumber of bytes what would be read. If multiple indexes share a name\nwith different arity it'll throw a 400.","operationId":"api_test_all_api_bio_all__index__head","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/bio/all/{index}/{arity}":{"head":{"tags":["bio"],"summary":"Api Test All Arity","description":"Query the database fetch ALL records for a given index and arity. Don't read\nthe records from S3, but instead set the Content-Length to the total\nnumber of bytes what would be read.","operationId":"api_test_all_arity_api_bio_all__index___arity__head","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"},{"required":true,"schema":{"title":"Arity","type":"integer"},"name":"arity","in":"path"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/bio/varIdLookup/{rsid}":{"get":{"tags":["bio"],"summary":"Api Lookup Variant For Rs Id","description":"Lookup the variant ID for a given rsID.","operationId":"api_lookup_variant_for_rs_id_api_bio_varIdLookup__rsid__get","parameters":[{"required":true,"schema":{"title":"Rsid","type":"string"},"name":"rsid","in":"path"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/bio/query/{index}":{"get":{"tags":["bio"],"summary":"Api Query Index","description":"Query the database for records matching the query parameter and\nread the records from s3.","operationId":"api_query_index_api_bio_query__index__get","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"},{"required":true,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"},{"required":false,"schema":{"title":"Fmt","default":"row"},"name":"fmt","in":"query"},{"required":false,"schema":{"title":"Limit","type":"integer"},"name":"limit","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}},"head":{"tags":["bio"],"summary":"Api Test Index","description":"Query the database for records matching the query parameter. Don't\nread the records from S3, but instead set the Content-Length to the\ntotal number of bytes what would be read. If the total number of\nbytes read exceeds a pre-configured server limit, then a 413\nresponse will be returned.","operationId":"api_test_index_api_bio_query__index__head","parameters":[{"required":true,"schema":{"title":"Index","type":"string"},"name":"index","in":"path"},{"required":true,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/bio/schema":{"get":{"tags":["bio"],"summary":"Api Schema","description":"Returns the GraphQL schema definition (SDL).","operationId":"api_schema_api_bio_schema_get","responses":{"200":{"description":"Successful Response","content":{"text/plain":{"schema":{"type":"string"}}}}}}},"/api/bio/query":{"post":{"tags":["bio"],"summary":"Api Query Gql","description":"Treat the body of the POST as a GraphQL query to be resolved.","operationId":"api_query_gql_api_bio_query_post","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}},"/api/bio/cont":{"get":{"tags":["bio"],"summary":"Api Cont","description":"Lookup a continuation token and get the next set of records.","operationId":"api_cont_api_bio_cont_get","parameters":[{"required":true,"schema":{"title":"Token","type":"string"},"name":"token","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/portal/groups":{"get":{"tags":["portal"],"summary":"Api Portal Groups","description":"Returns the list of portals available.","operationId":"api_portal_groups_api_portal_groups_get","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}},"/api/portal/restrictions":{"get":{"tags":["portal"],"summary":"Api Portal Restrictions","description":"Returns all restrictions for the current user.","operationId":"api_portal_restrictions_api_portal_restrictions_get","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}},"/api/portal/phenotypes":{"get":{"tags":["portal"],"summary":"Api Portal Phenotypes","description":"Returns all available phenotypes or just those for a given\ndisease group.","operationId":"api_portal_phenotypes_api_portal_phenotypes_get","parameters":[{"required":false,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/portal/complications":{"get":{"tags":["portal"],"summary":"Api Portal Complications","description":"Returns all available complication phenotype pairs.","operationId":"api_portal_complications_api_portal_complications_get","parameters":[{"required":false,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/portal/datasets":{"get":{"tags":["portal"],"summary":"Api Portal Datasets","description":"Returns all available datasets for a given disease group.","operationId":"api_portal_datasets_api_portal_datasets_get","parameters":[{"required":false,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/portal/documentation":{"get":{"tags":["portal"],"summary":"Api Portal Documentation","description":"Returns all available phenotypes or just those for a given\nportal group.","operationId":"api_portal_documentation_api_portal_documentation_get","parameters":[{"required":true,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"},{"required":false,"schema":{"title":"Group","type":"string"},"name":"group","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/portal/documentations":{"get":{"tags":["portal"],"summary":"Api Portal Documentations","operationId":"api_portal_documentations_api_portal_documentations_get","parameters":[{"required":true,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/portal/systems":{"get":{"tags":["portal"],"summary":"Api Portal Systems","description":"Returns system-disease-phenotype for all systems.","operationId":"api_portal_systems_api_portal_systems_get","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}},"/api/portal/links":{"get":{"tags":["portal"],"summary":"Api Portal Links","description":"Returns one - or all - redirect links.","operationId":"api_portal_links_api_portal_links_get","parameters":[{"required":false,"schema":{"title":"Q","type":"string"},"name":"q","in":"query"},{"required":false,"schema":{"title":"Group","type":"string"},"name":"group","in":"query"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/raw/plot/dataset/{dataset}/{file}":{"get":{"tags":["raw"],"summary":"Api Raw Plot Dataset","description":"Returns a raw, image plot for a dataset.","operationId":"api_raw_plot_dataset_api_raw_plot_dataset__dataset___file__get","parameters":[{"required":true,"schema":{"title":"Dataset","type":"string"},"name":"dataset","in":"path"},{"required":true,"schema":{"title":"File","type":"string"},"name":"file","in":"path"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/raw/plot/phenotype/{phenotype}/{file}":{"get":{"tags":["raw"],"summary":"Api Raw Plot Phenotype","description":"Returns a raw, image plot for the bottom-line analysis of a phenotype.","operationId":"api_raw_plot_phenotype_api_raw_plot_phenotype__phenotype___file__get","parameters":[{"required":true,"schema":{"title":"Phenotype","type":"string"},"name":"phenotype","in":"path"},{"required":true,"schema":{"title":"File","type":"string"},"name":"file","in":"path"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/api/raw/plot/phenotype/{phenotype}/{ancestry}/{file}":{"get":{"tags":["raw"],"summary":"Api Raw Plot Phenotype Ancestry","description":"Returns a raw, image plot for the bottom-line analysis of a phenotype.","operationId":"api_raw_plot_phenotype_ancestry_api_raw_plot_phenotype__phenotype___ancestry___file__get","parameters":[{"required":true,"schema":{"title":"Phenotype","type":"string"},"name":"phenotype","in":"path"},{"required":true,"schema":{"title":"Ancestry","type":"string"},"name":"ancestry","in":"path"},{"required":true,"schema":{"title":"File","type":"string"},"name":"file","in":"path"}],"responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}},"422":{"description":"Validation Error","content":{"application/json":{"schema":{"$ref":"#/components/schemas/HTTPValidationError"}}}}}}},"/":{"get":{"summary":"Index","description":"SPA demonstration page.","operationId":"index__get","responses":{"200":{"description":"Successful Response","content":{"application/json":{"schema":{}}}}}}}},"components":{"schemas":{"HTTPValidationError":{"title":"HTTPValidationError","type":"object","properties":{"detail":{"title":"Detail","type":"array","items":{"$ref":"#/components/schemas/ValidationError"}}}},"ValidationError":{"title":"ValidationError","required":["loc","msg","type"],"type":"object","properties":{"loc":{"title":"Location","type":"array","items":{"type":"string"}},"msg":{"title":"Message","type":"string"},"type":{"title":"Error Type","type":"string"}}}}}}""",
                                         verbose=True, prompt = """
                                         You will recieve a dictionary like this:
                                        "API Call prefix": "https://bioindex.hugeamp.org/api/bio/count/", "API Call Suffix": "&fmt=row", "index": "gene-associations", "q": "TREM2"
                                        Output a link like this as an API call, to check if the link works:
                                        https://bioindex.hugeamp.org/api/bio/count/gene-associations?q=TREM2&fmt=row
                                        Use gene-finder for questions like "What genes are associated with LateAD?"
                                        Use gene-associations for questions like "What phenotypes are associated with TREM2?"
                                        Use gene-expression for questions like "What tissues is TREM2 expressed in?"
                                        Use top-associations for variant questions like "What TREM2 variants are there?"
                                        Examples of more valid links:
                                        Use 
                                        https://bioindex.hugeamp.org/api/bio/query/gene-finder?q=LateAD&fmt=row
                                        https://bioindex.hugeamp.org/api/bio/count/gene-expression?q=TREM2&fmt=row
                                        https://bioindex.hugeamp.org/api/bio/count/top-associations?q=TREM2&fmt=row
                                        Then show the link in your answer to the user, replacing "count" with query " Data found at the following link: https://bioindex.hugeamp.org/api/bio/count/gene-associations?q=TREM2&fmt=row"
                                         
                                        """)
from typing import Optional, Dict, Union
from langchain.tools import StructuredTool
import json

import requests
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec

from langchain.agents import create_json_agent, AgentExecutor
def extract_data_as_json(json_object):
    extracted_data = json_object.get('data', [])
    return json.dumps({"data": extracted_data})
def convert_to_columnar(data_list):
    columnar_data = {}
    for record in data_list:
        for key, value in record.items():
            if key not in columnar_data:
                columnar_data[key] = []
            columnar_data[key].append(value)
    return columnar_data




def generate_api_url_and_analyze(api_params_str: str) -> str:
    try:
        # Parse the input string to a dictionary
        api_params = json.loads(api_params_str.replace("'", '"'))

        # Extract parameters and construct API URL
        prefix = api_params.get("API Call prefix", "")
        suffix = api_params.get("API Call Suffix", "")
        index = api_params.get("index", "")
        query = api_params.get("q", "")
        api_url = f"{prefix}{index}?q={query}{suffix}"
        display_url = api_url.replace("count", "query")

        # Construct the display URL
        return f"Data found at the following link: {display_url}"

    except Exception as e:
        return f"An error occurred: {e}"


# Sample usage (replace `json_agent_executor` with your actual executor)
# result = generate_api_url_and_analyze(test_params_str, test_question, json_agent_executor)
# print(result)



#Tool list - edit to give new capabilities to choose from. See Langchain docs on tools
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.utilities import PythonREPL
from langchain.tools import HumanInputRun

aipython=     PythonAstREPLTool()

tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
   # Tool(
   #     name="Pubmed Vector QA",
   #     description="useful for when you need to answer questions about research in neurodegenerative diseases. State your sources in the final answer",
   # ),
   Tool(
        name="Pubmed search",
        func=pubmedq,
        description="useful for when you need to answer questions about pubmed papers. Ask short questions i.e.  APOE and Alzheimers",
    ),
   StructuredTool.from_function(
        func=generate_api_url_and_analyze,
        name="Generate BioIndex API URL",
        description="""
Makes API calls to the hugeamp bioindex.
Tell hugeamp to report a link starting with https://bioindex.hugeamp.org/api/bio/count/?q= and ending in &fmt=row based on given question. 
Try to figure out the object and category of the question to find what q and index to use.
Also report the genes or phenotypes being asked about.
In the API, Alzheimer's is labelled AD, and Parkinson's is labelled Parkinsons.
Examples: "what phenotypes are associated with TREM2?":
"API Call prefix": "https://bioindex.hugeamp.org/api/bio/count/", "API Call Suffix": "&fmt=row", "index": "gene-associations", "q": "TREM2"
"What genes are associated with Alzheimer's?"
"API Call prefix": "https://bioindex.hugeamp.org/api/bio/count/", "API Call Suffix": "&fmt=row", "index": "gene-finder", "q": "AD"
"What TREM2 variants are there?"
"API Call prefix": "https://bioindex.hugeamp.org/api/bio/count/", "API Call Suffix": "&fmt=row", "index": "top-associations", "q": "TREM2"
"In what tissues is TREM2 most expressed?"
"API Call prefix": "https://bioindex.hugeamp.org/api/bio/count/", "API Call Suffix": "&fmt=row", "index": "gene-expression", "q": "TREM2"

Then show the link in your answer to the user

""",
    ),
    Tool(name="python REPL", func=aipython, description=""" Python shell. Use this to execute python commands.
            Input should be a valid python command.
            Don't ever use this right after using the Bioindex API.
            When using this tool, sometimes output is abbreviated -
            make sure it does not look abbreviated before using it in your answer
            If the previous question referenced an api link, use requests to download the json from the link and convert its 'data' dictionary to a csv. Then use pandas to answer the user's analysis question.
            NEVER print more then ten rows of data at once, you will cause an error."""),



]


#Prompt setup - changing this could be helpful
chat_history = ""
prefix = """Have a conversation with a human, answering the following questions as best you can.
Be quite verbose, documenting additional information, useful papers, and explanations. expand the names of abbreviations.
 Report exact values in lists when statistics come up.
 You have access to the following tools:"""
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


output_container = st.empty()

output_container = output_container.container()
import re
import requests

def fetch_and_display_json(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.write(f"Failed to get data from {url}")
        return

    raw_data = response.json()
    records = raw_data.get("data", [])  # Extract the list under the 'data' key

    # Sort the records by 'zStat' in descending order
    sorted_records = sorted(records, key=lambda x: x.get("zStat", 0), reverse=True)
    
    # Get the top 10 records (or all if there are fewer than 10)
    top_10_records = sorted_records[:10]
    # Display the top 10 records in a Streamlit table
    st.table(top_10_records)
    return records
def check_and_execute(string):
    # Use regular expression to find the URL pattern
    pattern = r"https://bioindex\.hugeamp\.org/api/bio/query[^\s.,!\]]*"
    match = re.search(pattern, string)
    
    if match:
        url = match.group(0)
        records = fetch_and_display_json(url)
        return records
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
        records =check_and_execute(st.session_state["generated"][i])

        print(st.session_state["generated"][i])
        #message(st.session_state['source'][i],  key=str(i))
        message(st.session_state['past'][i], is_user=True,
                avatar_style="big-ears",  key=str(i) + '_user')


