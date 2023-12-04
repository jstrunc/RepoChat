import os

import streamlit as st
from git import GitCommandError, Repo
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

from utils import (
    MyRetrievalQAWithSourcesChain,
    MyStrOutputParser,
    StreamingOutCallbackHandler,
    build_rag_chat_with_memory,
    create_document_db_from_repo,
)

api_key_default = os.environ["OPENAI_API_KEY"]
available_models = ("gpt-3.5-turbo-1106", "gpt-4-1106-preview")
default_repo_url = "https://github.com/IBM/ibm-generative-ai"

st.set_page_config(page_title="RepoChat", layout="wide")
st.title("RepoChat")

if "message_placeholder" not in st.session_state:
    st.session_state["message_placeholder"] = st.empty()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repo_url" not in st.session_state:
    st.session_state["repo_url"] = default_repo_url
    st.session_state["db"] = create_document_db_from_repo(default_repo_url)

if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None


if "model" not in st.session_state:
    openai_api_key = api_key_default
    model = available_models[0]

    st.session_state["model"] = available_models[0]
    st.session_state["streaming_out_callback"] = StreamingOutCallbackHandler(st.session_state["message_placeholder"])
    st.session_state["combine_llm"] = (
        ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=0.1,
            max_tokens=500,
            streaming=True,
            callbacks=[st.session_state["streaming_out_callback"]],
        )
        # | MyStrOutputParser()
    )
    st.session_state["partial_steps_llm"] = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=model,
        temperature=0.1,
        max_tokens=500
        # streaming=True
    )
    # , streaming=True)  # token counting with get_openai_callback() doesn't work with streaming=True

    st.session_state["memory"] = ConversationSummaryMemory(
        llm=st.session_state["partial_steps_llm"], memory_key="chat_history", output_key='answer', return_messages=True
    )

    if (
        st.session_state["db"]
        and st.session_state["combine_llm"]
        and st.session_state["partial_steps_llm"]
        and st.session_state["memory"]
    ):
        retriever = st.session_state["db"].as_retriever(
            # "mmr" Maximum Marginal Relevance - optimizes for similarity to query and diversity among selected documents
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        st.session_state["qa_chain"] = MyRetrievalQAWithSourcesChain.from_2llms(
            question_llm=st.session_state["partial_steps_llm"],
            combine_llm=st.session_state["combine_llm"],
            retriever=retriever,
            memory=st.session_state["memory"],
            return_source_documents=True,
        )

        # add parser to format the output text into html
        # st.session_state["qa_chain"] =

        # TODO: Add a formatter to the chain to display source documents as clickable web links in markdown format
        # and display cosine similarity of the sources, optionally add a popup with relevant text found
        # Add controls for specifying the number of retrieved documents, token limit, ...

        # TODO: Add the ability to perform Google/DuckDuckGo searches

        # TODO: Add input format validation
        # TODO: Add a collapsible menu in the sidebar with information about the application, possibly about the repository, and a settings tab

        # TODO: Host on Azure

assistant_identity = (
    'You are a top AI researcher from OpenAI, ' 'previously working at Google Brain with PhD from Stanford'
)

with st.sidebar.form(key='model_form'):
    st.markdown("## Assistant settings")
    openai_api_key = st.text_input('OpenAI API key:', value=api_key_default if api_key_default else '')
    st.text_area('Assistant identity:', value=assistant_identity, height=30)
    model = st.selectbox('Model:', available_models, index=0)

    if st.form_submit_button(label='Re/load model'):
        pass

        # TODO: It is not necessary to delete messages, just create a new qa_chain with a new model


with st.sidebar.form(key='repo_url_form'):
    st.markdown("## Repository")
    repo_url = st.text_input('URL:', value=default_repo_url)

    if st.form_submit_button(label='Load repository') and repo_url != st.session_state["repo_url"]:
        # check if repo url exists
        repo_path = f"/tmp/repo_chat/{repo_url.split('/')[-1]}"
        repo = None
        try:
            repo = Repo.clone_from(repo_url, to_path=repo_path)
        except GitCommandError as e:
            repo_not_found_msg = "Repository not found"
            if repo_not_found_msg in e.stderr:
                st.error(f"{repo_not_found_msg} at URL: {repo_url}\nFix the URL and try again.")

        if repo:
            st.session_state["qa_chain"] = None

            st.session_state["messages"] = []
            st.session_state["repo_url"] = repo_url
            st.markdown(f"Loading repository from {repo_url}")

    else:
        st.markdown("This repository is already loaded.")

if st.session_state["qa_chain"]:
    st.sidebar.success(f"Repository {st.session_state['repo_url']} loaded")
    st.sidebar.success(f"Model {st.session_state['model']} loaded")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about {st.session_state['repo_url']}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.session_state["streaming_out_callback"].output_streamlit_placeholder = st.empty()
            with st.spinner("Retrieving documents from vector store & composing the answer..."):
                # prompt = "Which methods has the Credentials class?"
                result = st.session_state["qa_chain"].invoke(prompt)
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})
        print(result)

else:
    st.sidebar.error(f"Repository {st.session_state['repo_url']} not loaded")
    st.sidebar.error(f"Model {st.session_state['model']} not loaded")
    st.chat_input("Load a repository first", disabled=True)
