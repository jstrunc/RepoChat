# RepoChat

RepoChat is an interactive chatbot application that uses OpenAI's latest cost efficient models (GPT-3.5-turbo-1106 and GPT-4-1106-preview) to answer questions about a given public Git repository and corresponding online documentation. It clones the repository, parses the documentation from the web, creates a local Chroma vector database and then uses custom LangChain RAG chain to answer questions based on that database.

## Features

- **Interactive Chat Interface**: RepoChat uses Streamlit to create an interactive chat interface where users can ask questions and get answers (and ask followup questions!).
- **OpenAI Models**: RepoChat currently uses OpenAI's models, but can be extended to any other provider supported through the LangChain interface.
- **Document Database**: RepoChat creates a document database from a given GitHub repository and online documentation. This database is used to provide context to the OpenAI models and improve the accuracy of their responses. 
- **Jump to Information Source**: The files identified as sources of information relevant to the question (used by the models to generate the answer) are provided together with the answers in the form of clickable links.
- **Inspect Source Documents Details**: each answer includes expandable container with all document chunks retrieved from the vector DB that are most relevant to the question including their similarity score, content and source file name.


## Installation

1. Clone the RepoChat repository.
2. Create a virtual environment e.g. using conda: `conda create --name repochat python=3.11`
3. Activate the virtual environment: `conda activate repochat`
4. Install poetry (used for management of dependencies): `pip install poetry`
4. Install the required Python packages (the exact versions from poetry.lock) without the project itself using poetry: `poetry install --no-root`
3. Set the `OPENAI_API_KEY` environment variable to your OpenAI API key (or set it later from the Streamlit application UI).
4. Run the Streamlit app (opens in a new tab in your default browser): `streamlit run streamlit_app.py`

## Usage

1. In the sidebar tab Settings:
    1. Select model, enter your OpenAI API key (if not provided as the `OPENAI_API_KEY` environment variable before running the app) and click the "Re/load model" button.
    2. Enter the URL of a Git repository (tested only on GitHub url's) in the "Repository URL" field.
    3. Optionally enter URL of the index (or contents page) the online documentation (all pages linked from this page will be parsed, chunked and stored in the vector DB).
    4. Click the "Load Repository" button to create a document database from the repository.
2. Enter your questions in the prompt field at the bottom and press Enter to get a response from the chatbot.
3. For more details about the answer expand the "Full result" container bellow the answer.
4. The model can be changed during the conversation (if weaker models answer isn't good enough, switch to stronger (and more expensive) model to answer using the same chat history).

## Limitations

- currently only supports following files in the repository: .py, .md

## Contact

If you have any questions or feedback, please contact the project maintainers:

- [Josef Strunc](mailto:josef.strunc@gmail.com)