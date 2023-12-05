"""Langchain functionality"""

import os
from typing import Any

import pandas as pd
from git import Repo
from langchain.chains import ReduceDocumentsChain, RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.schema import BasePromptTemplate
from langchain.schema.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import BaseTransformOutputParser, StrOutputParser
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate

def create_document_db_from_repo(repo_url: str) -> VectorStore:
    repo_path = f"/tmp/repo_chat/{repo_url.split('/')[-1]}"
    chroma_path = repo_path + "/chroma_db"

    # create new DB with embeddings
    if not os.path.exists(chroma_path) or 'chroma.sqlite3' not in os.listdir(chroma_path):
        if os.path.exists(repo_path) and (repo_content := os.listdir(repo_path)) and '.git' in repo_content:
            print(f"Repo already exists at {repo_path}")
            repo = Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            repo = Repo.clone_from(repo_url, to_path=repo_path)

        # Load
        loader = GenericLoader.from_filesystem(
            repo_path + "/src/genai",
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        )
        documents = loader.load()
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON, chunk_size=1000, chunk_overlap=200
        )
        print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))
        texts = python_splitter.split_documents(documents)

        db = Chroma.from_documents(
            texts,
            OpenAIEmbeddings(disallowed_special=()),
            persist_directory=chroma_path,
            collection_metadata={"hnsw:space": "cosine"},
        )
        db.persist()

    else:
        # load existing DB
        db = Chroma(
            embedding_function=OpenAIEmbeddings(disallowed_special=()),
            persist_directory=chroma_path,
            collection_metadata={"hnsw:space": "cosine"},
        )

    return db


def build_rag_chat_with_memory(openai_api_key: str, model: str, db: VectorStore, output_component) -> Chain:
    retriever = db.as_retriever(
        # "mmr" Maximum Marginal Relevance - optimizes for similarity to query and diversity among selected documents
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    callbacks_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=model,
        streaming=True,
        temperature=0.1,
        max_tokens=500,
        callbacks=[StreamingOutCallbackHandler(output_component)],
    )
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model, streaming=True, temperature=0.1, max_tokens=500)
    # , streaming=True)  # token counting with get_openai_callback() doesn't work with streaming=True

    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", output_key='answer', return_messages=True)
    qa = MyRetrievalQAWithSourcesChain.from_2llms(
        question_llm=llm, combine_llm=callbacks_llm, retriever=retriever, memory=memory, return_source_documents=True
    )

    return qa


class StreamingOutCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(
        self,
        output_streamlit_placeholder,
        repo_url: str = "https://github.com/IBM/ibm-generative-ai",
        repo_path: str = "/tmp/repo_chat/ibm-generative-ai",
    ):
        self.output_streamlit_placeholder = output_streamlit_placeholder
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.final_message = None

        self._message_buffer = ""
        self._source_buffer = ""
        self._sources = []
        self._parsing_message = True

    @property
    def _formatted_sources(self) -> str:
        """String in streamlit markdown format with sources in a bullet list with clickable links."""
        sources_bullet_list = "\n\n - ".join([f"[{source}]({source})" for source in self._sources])
        sources_bullet_list = f"\n\n - {sources_bullet_list}" if sources_bullet_list else ""
        return sources_bullet_list

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Parses each new incoming token and prints formatted message to the streamlit output placeholder.

        Runs on new LLM token. Only available when streaming is enabled.
        Identified document sources are transformed into URLs and formatted as a bullet list.
        """
        if self._parsing_message:
            self._message_buffer += token
            if "\nSOURCES:" in self._message_buffer:
                self._message_buffer = self._message_buffer.replace("\nSOURCES:", "")
                self._parsing_message = False
        else:  # parsing sources
            if token == ",":  # append complete source url - change local repo path to url
                # TODO: fix the "/tree/main": this will probably only work on Github
                self._sources.append(self._source_buffer.lstrip().replace(self.repo_path, f"{self.repo_url}/tree/main"))
                self._source_buffer = ""
            else:  # continue parsing single source
                self._source_buffer += token

        streamed_message = self._message_buffer + "\n\nSources:" if (self._sources or self._message_buffer) else ""
        streamed_message += self._formatted_sources
        streamed_message += f"\n\n - {self._source_buffer}" if self._source_buffer else ""
        self.output_streamlit_placeholder.markdown(streamed_message)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        # source still in buffer:
        if self._source_buffer:
            self._sources.append(self._source_buffer.lstrip().replace(self.repo_path, f"{self.repo_url}/tree/main"))

        self.final_message = self._message_buffer + f"\n\nSources: {self._formatted_sources}" if self._sources else ""
        self.output_streamlit_placeholder.markdown(self.final_message)
        # self.output_streamlit_placeholder.markdown(
        #     pd.DataFrame(self._sources).to_html(render_links=True), unsafe_allow_html=True
        # )

        # reset internal variables for the next run
        self._message_buffer = ""
        self._source_buffer = ""
        self._sources = []
        self._sources_links = []
        self._parsing_message = True


class MyStrOutputParser(StrOutputParser):
    """OutputParser that parses LLMResult into the top likely string."""

    def parse(self, result: dict) -> str:
        """Returns the input text with no changes."""
        return result['answer']


repo_combine_prompt_template = """{assistant_identity}\n
Given the chat history, the question and the extracted parts of multiple source files with programming code, all from one repository, create a final answer with references ("SOURCES"). 
If there is no relevant information in the sources summaries, look for it in the last CHAT HISTORY.
If you still don't know the answer, just say that you don't know. Don't try to make up an answer.
If the QUESTION is not an actual question, just a statement, just answer accordingly, e.g.: "Ok.", "Got it.", "Thank you." or "Sure.", ignore the given extracted content.
ALWAYS return a "SOURCES" part in your answer.

CHAT HISTORY: The user asks which classes are available that are derived from BaseMessage. The assistant responds with: "The available child classes of BaseMessage are: HumanMessage, SystemMessage and AIMessage."
=========
QUESTION: Which class is used for holding the parameters for listing tunes?
=========
Content: from typing import Optional\n
from pydantic import BaseModel, ConfigDict, Field\n
# TODO: Update the descriptions import
from genai.schemas.descriptions import TunesAPIDescriptions as tx\n\n
class TunesListParams(BaseModel):
    \"\"\"Class to hold the parameters for listing tunes.\"\"\"\n
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")\n
    limit: Optional[int] = Field(None, description=tx.LIMIT, le=100)
    offset: Optional[int] = Field(None, description=tx.OFFSET)
Source: /tmp/repo_chat/ibm-generative-ai/src/genai/schemas/tunes_params.py
Content: from typing import Optional\n
from pydantic import BaseModel, ConfigDict, Field\n
from genai.schemas.descriptions import FilesAPIDescriptions as tx\n\n
class FileListParams(BaseModel):
    \"\"\"Class to hold the parameters for file listing.\"\"\"\n
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")\n
    limit: Optional[int] = Field(None, description=tx.LIMIT, le=100)
    offset: Optional[int] = Field(None, description=tx.OFFSET)
Source: /tmp/repo_chat/ibm-generative-ai/src/genai/schemas/files_params.py
=========
FINAL ANSWER: Class used for holding the parameters for listing tunes is TunesListParams.
SOURCES: /tmp/repo_chat/ibm-generative-ai/src/genai/schemas/tunes_params.py


CHAT HISTORY: {chat_history}
=========
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
REPO_COMBINE_PROMPT = PromptTemplate(
    template=repo_combine_prompt_template,
    input_variables=["assistant_identity", "summaries", "question", "chat_history"],
)


class RepoRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):
    """Custom repository question answering chain summarizing the question-relevant parts of retrieved documents.

    Uses one LLM to summarize only the question-relevant parts of the retrieved code chunks and another to combine
    the question, chat history and the summaries into a final answer. The combine LLM prompt contains a single shot
    example. The combine LLM is also personalized with the assistant identity.
    """

    # output_parser: BaseTransformOutputParser = MyStrOutputParser()

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.output_parser = StrOutputParser()

    @classmethod
    def from_llms(
        cls,
        assistant_identity: str,
        question_llm: BaseLanguageModel,
        combine_llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = REPO_COMBINE_PROMPT,
        **kwargs: Any,
    ) -> BaseQAWithSourcesChain:
        """Construct the chain from an LLM."""
        combine_prompt = combine_prompt.partial(assistant_identity=assistant_identity)
        llm_question_chain = LLMChain(llm=question_llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=combine_llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_results_chain)
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",
        )
        return cls(
            combine_documents_chain=combine_documents_chain,
            **kwargs,
        )

    # def process(self, text: str) -> str:
    #     return self.output_parser.parse(text)
