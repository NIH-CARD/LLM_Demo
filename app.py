import os
import openai
from PIL import Image
from streamlit_chat import message
from utils import *

openai.api_key = st.secrets["openai_api_key"]
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
st.header("CARD.AI")


def chat_query(query):
    # start chat with chatOutside
    try:
        response = qa_with_sources(query)
        answer = response['answer']
        source = response['sources']

    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    if len(answer) > 0:
        return answer, source

    else:
        return None
# ============================================================


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

if user_input:
    # source contain urls from Outside
    output, source = chat_query(user_input)

    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.source.append(source)

    # Display source urls
    st.write(source)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i],  key=str(i))
        #message(st.session_state['source'][i],  key=str(i))
        message(st.session_state['past'][i], is_user=True,
                avatar_style="big-ears",  key=str(i) + '_user')
       