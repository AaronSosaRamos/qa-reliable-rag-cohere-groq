### Build Index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv, find_dotenv
from utils.logger import setup_logger

load_dotenv(find_dotenv())

logger = setup_logger(__name__)

# Set embeddings
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

def return_retriever(urls):
  # Load
  docs = [WebBaseLoader(url).load() for url in urls]
  docs_list = [item for sublist in docs for item in sublist]

  # Split
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=500, chunk_overlap=0
  )
  doc_splits = text_splitter.split_documents(docs_list)

  logger.info("Generating Vector Store with Cohere and URLs")

  # Add to vectorstore
  vectorstore = Chroma.from_documents(
      documents=doc_splits,
      collection_name="rag",
      embedding=embedding_model,
  )

  logger.info("Vector Store successfully generated")
  logger.info("Generating retriever")

  retriever = vectorstore.as_retriever(
                  search_type="similarity",
                  search_kwargs={'k': 4}, # number of documents to retrieve
              )
  
  logger.info("Retriever successfully generated")
  
  return retriever