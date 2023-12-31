import os
import re
import sys
from urllib.parse import urlparse

import requests
import streamlit as st

from repo_rag_chat import RepoRagChatAssistant

# TODO: Input fields for the number of retrieved documents, token limit, ...
# TODO: Google/DuckDuckGo searches


def url_valid(url: str) -> bool:
    """Validates the URL by checking if it has a correct format and if it is reachable returning 200 status code."""
    try:
        if url:
            parsed_url = urlparse(url)
            return parsed_url.scheme and parsed_url.netloc and requests.get(url).status_code == 200
    except Exception as e:
        print(f"Error validating URL {url}: {e}", file=sys.stderr)
    return False


def change_label_style(label, font_size='12px'):
    """Helper function to change label style"""
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
    </script>
    """
    st.components.v1.html(html, height=0)


api_key_default = os.environ.get("OPENAI_API_KEY", default="")
available_models = ("gpt-3.5-turbo-1106", "gpt-4-1106-preview")
default_repo_url = "https://github.com/IBM/ibm-generative-ai"
default_documentation_url = "https://ibm.github.io/ibm-generative-ai/index.html"
default_repo_local_path = "/tmp/repo_chat"
default_num_documents_to_retrieve = 4
initial_message = {
    "role": "assistant",
    "content": "Hi, I'm RepoChat. I can answer questions about any code repository.",
}
default_assistant_identity = (
    "You are a top ex Google software engineer, you are really good at understanding code and answering questions"
    " about any code repository."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [initial_message]

if "default_model_tried" not in st.session_state:
    st.session_state["default_model_tried"] = False

if "default_repo_tried" not in st.session_state:
    st.session_state["default_repo_tried"] = False


# Start building UI
st.set_page_config(page_title="RepoChat", layout="wide")
st.title("RepoChat")

if "message_placeholder" not in st.session_state:
    st.session_state["message_placeholder"] = st.empty()

if "assistant" not in st.session_state:
    st.session_state["assistant"] = RepoRagChatAssistant(st.session_state["message_placeholder"])

sidebar_tab_settings, sidebar_tab_about = st.sidebar.tabs(
    # hacky way to change font size - use latex: rf"$\textsf{{\normalsize {label}}}$"
    # sizes: \Huge \huge \LARGE \Large \large \normalsize \small \footnotesize
    [r"$\textsf{\large Settings}$", r"$\textsf{\large About}$"]
)

with sidebar_tab_about:
    st.markdown(
        "### RepoChat is a conversational AI assistant that can answer questions about code repositories.\n\n "
        "## Features: \n"
        "- **Conversational** - you can ask multiple questions in one conversation\n"
        "- **Context-aware** - the assistant remembers the conversation history and can use it to answer questions\n"
        "- **Retrieval augmented generation** - the assistant uses a vector store to find relevant information\n"
        "- **Jump to information source** - The files identified as sources of information relevant to the question"
        " are provided together with the answers in the form of clickable links\n"
        "- **Open-domain** - the assistant can answer questions about any code repository\n"
        "- **Open-ended** - the assistant can answer questions that are not explicitly mentioned in the repository\n\n"
        "Limitations:\n - Currently only supports .py and .md files\n\n"
        "## Author: \n[Josef Strunc](mailto:josef.strunc@gmail.com)"
    )

# Loading new model:
# 1) Read inputs from the UI
# 2) Load the model (creates Assistant's LLMs and memory) on first run with default values or on button click
#     - only if the API key is provided (model is also selected valid from the dropdown and assistant_identity
#     can be anything for now)
#     - if retriever already exists: creates new qa_chain in the assistant
#     - if memory already exists it is reused
with sidebar_tab_settings.form(key="assistant_form"):
    st.markdown("## Assistant")
    openai_api_key = st.text_input("OpenAI API key:", value=api_key_default)
    assistant_identity = st.text_area(
        "Assistant identity:",
        value=default_assistant_identity,
        height=120,
        help="Type any description of \"personality\" of the assistant. "
        "\n\nThis text is prepended to every internally constructed prompt.",
    )
    model = st.selectbox("Model:", available_models, index=0)  # if SB is edited, no option is selected, it returns None

    load_model_clicked = st.form_submit_button(
        label="Load model",
        help="Loads the model with the settings above. \n\nKeeps any existing conversation in memory ",
    )
    # on submit button click or on initial load (loading default model)
    if openai_api_key and model and (load_model_clicked or not st.session_state["default_model_tried"]):
        with st.spinner("Loading model..."):
            result = st.session_state["assistant"].load_model(openai_api_key, model, assistant_identity)
        if result == RepoRagChatAssistant.SUCCESS_MSG:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"I am now using the model **{model}** internally."}
            )
        else:
            st.error(result)
    if st.session_state["assistant"].model:
        st.success(f"Model **{st.session_state['assistant'].model}** loaded.")

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
with sidebar_tab_settings.form(key="repo_url_form"):
    st.markdown("## Repository")
    repo_url = st.text_input("URL:", value=default_repo_url)

    documentation_url = st.text_input(
        "Documentation URL (optional):",
        value=default_documentation_url,
        help="Optional link to the documentation of the repository.\n\n"
        "All links from this URL will be parsed and added to the vector DB",
    )
    # with st.expander("Advanced Settings", expanded=False):    # use the expander when there is more detailed settings
    repo_local_path = st.text_input(
        "Local folder:",
        value=default_repo_local_path,
        help="Absolute path to folder where the repository will be cloned and the vector database stored.",
    )
    num_documents_to_retrieve = int(
        st.text_input(
            "Number of document chunks to retrieve from the vector DB per answer:",
            value=default_num_documents_to_retrieve,
        )
    )
    load_repository_clicked = st.form_submit_button(
        label="Load repository",
        help="Loads the repository on the given URL.\n\nAny existing conversation is deleted ",
    )

    both_urls_valid = True
    if load_repository_clicked:
        if (
            st.session_state["assistant"].db
            and st.session_state["assistant"].num_documents_to_retrieve != num_documents_to_retrieve
        ):
            # if repo/DB is already loaded and num_documents_to_retrieve changed just re-create the retriever
            st.session_state["assistant"].create_retriever(num_documents_to_retrieve)

        if not url_valid(repo_url):
            st.error("Invalid or unreachable repository URL")
            both_urls_valid = False
        if documentation_url and not url_valid(documentation_url):
            st.error("Invalid documentation URL")
            both_urls_valid = False

    if (load_repository_clicked or not st.session_state["default_repo_tried"]) and both_urls_valid:
        with st.spinner("Loading repository..."):
            message = st.session_state["assistant"].load_repo(
                repo_url, repo_local_path, documentation_url, num_documents_to_retrieve
            )
        if message == RepoRagChatAssistant.SUCCESS_MSG:
            if st.session_state["default_repo_tried"]:
                st.session_state.messages = [initial_message]
            assistant_msg = f"I loaded the repository from {repo_url}"
            if documentation_url:
                assistant_msg += f" and the documentation from {documentation_url}"
            st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        else:
            st.markdown(message)

    if both_urls_valid and st.session_state["assistant"].repo_url:
        st.success(f"Repository **{st.session_state['assistant'].repo_url}** loaded.")

    st.session_state["default_repo_tried"] = True


# If the assistant's qa_chain is initialized (both repo and model are loaded), we can start the conversation
if st.session_state["assistant"].qa_chain:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "full_result" in message:
                docs_with_similarity = st.session_state["assistant"].similarity_search_with_score(
                    message["full_result"]["question"]
                )

                with st.expander("Full result (sources and similarity scores)"):
                    st.markdown(f"### Documents used for generating the final answer: \n")
                    # st.markdown("- ".join([doc for doc in list(message['full_result']['sources'])]))
                    st.markdown(message['full_result']['sources'])
                    st.markdown(
                        "### All document chunks retrieved from the vector DB with contents and similarity scores.\n"
                        "(Sorted, the most similar on the top. Lower score represents more similarity.)"
                    )
                    for i, doc in enumerate(message["full_result"]["source_documents"]):
                        # double check that db.similarity_search_with_score returned the same docs in the same order
                        assert doc.metadata["source"] == docs_with_similarity[i][0].metadata["source"]
                        st.markdown(
                            f"##### {i+1}. Score: {docs_with_similarity[i][1]:.4f}: \n**{doc.metadata['source']}**"
                        )
                        page_content_compacted = re.sub(r'\n+', '\n', doc.page_content)  # remove duplicated newlines

                        # TODO: better print of the documentation chunks - they are not all code
                        st.code(page_content_compacted)

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
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": st.session_state["assistant"].last_formatted_message,
                "full_result": result,
            }
        )

        st.rerun()  # to show the active chat_input again

else:
    st.chat_input("Load a repository first", disabled=True)
