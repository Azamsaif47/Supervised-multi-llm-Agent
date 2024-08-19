import os
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set the environment variable for your OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-proj-zgXqKiXJ3flv7xFoBvdjT3BlbkFJ1hezS1kRuOsw0jTkBEjE"
openai.api_key = os.getenv('OPENAI_API_KEY')

def load_and_split_pdfs(pdf_paths):
    all_pages = []

    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path, extract_images=False)  # Disable image extraction
            pages = loader.load()
            all_pages.extend(pages)  # Add the loaded pages to the list
            print(f"Successfully processed {pdf_path}.")
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    return all_pages

# Specify the folder containing the PDF files
folder_path = "pdf_files"

# Get a list of all PDF files in the folder
pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

# Load and split the PDFs
loaded_pages = load_and_split_pdfs(pdf_paths)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(loaded_pages)

persist_directory = "D:/azam_voice/chromadb/faiss.index"

embeddings = OpenAIEmbeddings()

# Create FAISS vector store
vectordb = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The notes the chunk is from, should be one of `1.pdf`, `2.pdf`, or `3.pdf` upto 19.pdf",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the research notes",
        type="integer",
    ),
]

document_content_description = "research notes"
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)  # Update the model name as needed

retriever = vectordb.as_retriever()
"""prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
"""


pdf_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Ensure the input has a 'query' key
def pdf_agent_invoke(inputs):
    query = inputs.get("query")
    if not query:
        raise ValueError("Missing input key: 'query'")
    return pdf_agent({"query": query})

# Example usage
question = "What is Declaration of Competing Interest?"  # Replace with your question
response = pdf_agent.invoke(question)
print(f"Response: {response}")