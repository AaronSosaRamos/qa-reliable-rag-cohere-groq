from schemas.schemas import GradeDocuments, GradeHallucinations, HighlightDocuments
from vector_store_db.vector_store_db import return_retriever
from dotenv import load_dotenv, find_dotenv
from utils.logger import setup_logger
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv(find_dotenv())

logger = setup_logger(__name__)
llm_llama_3_1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def generate_retriever(state):
    retriever = return_retriever(state['urls'])

    logger.info(f"1st node - Retriever successfully generated with {len(state['urls'])} URLs")

    return {
        "retriever": retriever
    }

def filter_doc(question, doc, retrieval_grader):
    """Function to grade a single document for relevance."""
    print(doc.page_content, '\n', '-'*50)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res, '\n')
    return doc if res.binary_score == 'yes' else None

def filter_non_relevant_docs(state):
    docs = state['retriever'].invoke(state['question'])
    # LLM with function call
    structured_llm_grader = llm_llama_3_1.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    docs_to_use = []

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        # Map each document to the grading function
        results = list(executor.map(lambda doc: filter_doc(state['question'], doc, retrieval_grader), docs))

    # Filter out None values (non-relevant documents)
    docs_to_use = [doc for doc in results if doc is not None]

    logger.info(f"2nd node - Filtered out non-relevant documents: {docs_to_use}")

    return {
        "docs_to_use": docs_to_use
    }

def format_docs(docs):
      return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

def generate_result(state):
  # Prompt
  system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
  Use three-to-five sentences maximum and keep the answer concise."""
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>."),
      ]
  )

  # Chain
  rag_chain = prompt | llm_llama_3_1 | StrOutputParser()

  logger.info(f"3rd node - Generated result for user question: {state['question']}")

  # Run
  generation = rag_chain.invoke({"documents":format_docs(state['docs_to_use']), "question": state['question']})
  return {
      "generation": generation
  }

def check_for_hallucinations(state):
  # LLM with function call
  structured_llm_grader = llm_llama_3_1.with_structured_output(GradeHallucinations)

  # Prompt
  system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
      Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
  hallucination_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
      ]
  )

  hallucination_grader = hallucination_prompt | structured_llm_grader

  response = hallucination_grader.invoke({"documents": format_docs(state['docs_to_use']), "generation": state['generation']})

  logger.info(f"4th node - Checked for hallucinations in the generated answer: {response.binary_score}")

  return {
      "is_grounded": response.binary_score
  }

def highlight_docs(state):
  # LLM
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

  # parser
  parser = JsonOutputParser(pydantic_object=HighlightDocuments)

  # Prompt
  system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
  1. A question.
  2. A generated answer based on the question.
  3. A set of documents that were referenced in generating the answer.

  Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
  generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
  in the provided documents.

  Ensure that:
  - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
  - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
  - (Important) If you didn't used the specific document don't mention it.

  Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

  You must respond as a JSON format:

  <format_instruction>
  {format_instructions}
  </format_instruction>
  """


  prompt = PromptTemplate(
      template= system,
      input_variables=["documents", "question", "generation"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )

  # Chain
  doc_lookup = prompt | llm | parser

  # Run
  lookup_response = doc_lookup.invoke({"documents":format_docs(state['docs_to_use']), "question": state['question'], "generation": state['generation']})

  logger.info(f"5th node - Highlighted relevant segments from the documents: {lookup_response}")

  return {
      "lookup_response": lookup_response
  }