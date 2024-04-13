import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# llm = Ollama(base_url='http://119.74.32.2:11434',
#              model="llama2", temperature=0)
llm = ChatOpenAI(model="gpt-3.5-turbo",
                 api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# embeddings_model = OpenAIEmbeddings(api_key="OPENAI_API_KEY")
embeddings = HuggingFaceEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=4000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)


def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path, extract_images=True)
    pages = loader.load_and_split(text_splitter=text_splitter)
    return pages


def generate_summary(texts):
    prompt_text = """You are an assistant tasked with summarising text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text elements. \
    Give a concise summary of the text that is well optimized for retrieval.
    Do not explain why the text is important, just summarize it.
    Text: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = {
        "element": lambda x: x} | prompt | llm | StrOutputParser()

    return summarize_chain.batch(texts, {"max_concurrency": 5})


def test_summaries():
    pages = load_and_split_pdf("./style.pdf")
    texts = [page.page_content for page in pages]
    summaries = generate_summary(texts)

    print(summaries)


# db = FAISS.from_documents(pages, embeddings)

# loader = UnstructuredPDFLoader("./style.pdf")
# data = loader.load()
# print(data)
