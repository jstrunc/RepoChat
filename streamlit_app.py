import os

import streamlit as st

from utils import RepoRagChatAssistant

api_key_default = os.environ.get("OPENAI_API_KEY", default="")
available_models = ("gpt-3.5-turbo-1106", "gpt-4-1106-preview")
default_repo_url = "https://github.com/IBM/ibm-generative-ai"
default_assistant_identity = (
    "You are a top ex Google software engineer, you are really good at understanding code and answering questions"
    " about any code repository."
)

st.set_page_config(page_title="RepoChat", layout="wide")
st.title("RepoChat")

if "message_placeholder" not in st.session_state:
    st.session_state["message_placeholder"] = st.empty()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state:
    st.session_state["assistant"] = RepoRagChatAssistant(st.session_state["message_placeholder"])

if "default_model_tried" not in st.session_state:
    st.session_state["default_model_tried"] = False

if "default_repo_tried" not in st.session_state:
    st.session_state["default_repo_tried"] = False

# add parser to format the output text into html

# TODO: Add a formatter to the chain to display source documents as clickable web links in markdown format
# and display cosine similarity of the sources, optionally add a popup with relevant text found
# Add controls for specifying the number of retrieved documents, token limit, ...

# TODO: Add the ability to perform Google/DuckDuckGo searches

# TODO: Add input format validation
# TODO: Add a collapsible menu in the sidebar with information about the application, possibly about the repository, and a settings tab

# TODO: Host on Azure

# Loading new model:
# 1) Read inputs from the UI
# 2) Load the model (creates Assistant's LLMs and memory) on first run with default values or on button click
#     - only if the API key is provided (model is also selected valid from the dropdown and assistant_identity
#     can be anything for now)
#     - if retriever already exists: creates new qa_chain in the assistant
#     - if memory already exists it is reused
with st.sidebar.form(key='model_form'):
    st.markdown("## Assistant settings")
    openai_api_key = st.text_input('OpenAI API key:', value=api_key_default)
    assistant_identity = st.text_area('Assistant identity:', value=default_assistant_identity, height=30)
    model = st.selectbox('Model:', available_models, index=0)

    # on submit button click or on initial load (loading default model)
    if openai_api_key and (st.form_submit_button(label='Re/load model') or not st.session_state["default_model_tried"]):
        result = st.session_state["assistant"].load_model(openai_api_key, model, assistant_identity)
        if result != RepoRagChatAssistant.SUCCESS_MSG:
            st.error(result)
    if st.session_state["assistant"].model:
        st.success(f"Model {st.session_state['assistant'].model} loaded")

    st.session_state["default_model_tried"] = True


# Loading new repository:
# 1) Read inputs from the UI
# 2) If the repo url is different than the one currently loaded in the Assistant:
#    - loads new repo from existing local folder or clones new repo
#    - creates new vector db or loads existing one if it exists
#    - creates new retriever
#    - creates new memory object - any existing conversation about different repo is not relevant
#    - if the model was loaded (LLMs already exist, API key works) - creates the final qa_chain in Assistant
# 3) Informs about the un/success of loading repo (creating vector db)
with st.sidebar.form(key='repo_url_form'):
    st.markdown("## Repository")
    repo_url = st.text_input('URL:', value=default_repo_url)

    if st.form_submit_button(label='Load repository') or not st.session_state["default_repo_tried"]:
        if repo_url != st.session_state["assistant"].repo_url:
            message = st.session_state["assistant"].load_repo(repo_url)
            if message == RepoRagChatAssistant.SUCCESS_MSG:
                st.session_state.messages = []
            else:
                st.error(message)
        else:
            st.markdown("This repository is already loaded.")

    if st.session_state["assistant"].repo_url:
        st.success(f"Repository {st.session_state['assistant'].repo_url} loaded")

    st.session_state["default_repo_tried"] = True


# If the assistant's qa_chain is initialized (both repo and model are loaded), we can start the conversation
if st.session_state["assistant"].qa_chain:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about {st.session_state['assistant'].repo_url}", key="chat_input"):
        # hacky way to disable the chat input while the assistant is working
        st.chat_input("Wait until the answer is generated...", key="disabled_chat_input", disabled=True)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.session_state["assistant"].output_callback.streamlit_output_placeholder = st.empty()
            with st.spinner("Retrieving documents from vector DB & composing the answer..."):
                result = st.session_state["assistant"](prompt)
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})

        st.rerun()  # to show the active chat_input again

else:
    st.chat_input("Load a repository first", disabled=True)
