from flask_cors import CORS
from flask import Flask, request
import glob
from functools import reduce

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

pdf_files = glob.glob(
    '/Users/aaron/Documents/Design Thinking/*.pdf', recursive=True)
# print(pdf_files)

img_files = glob.glob('/Users/aaron/Documents/**/*.jpg', recursive=True)
# print(img_files)

# embeddings_model = OpenAIEmbeddings(api_key="OPENAI_API_KEY")
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2")

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)


def init_db():
    dbs = []
    for file in pdf_files:
        try:
            loader = UnstructuredPDFLoader(file, extract_images=True)
            print(f"Loading {file}")
            pages = loader.load()
            # pages = loader.load_and_split(text_splitter=text_splitter)
            dbs.append(FAISS.from_documents(pages, embedding_function))
        except Exception as error:
            print(f"Error loading file {file}")
            print(error)
            continue
    db_merged = reduce(lambda db1, db2: db2.merge_from(db1)
                       if db2 is not None else db1, dbs)
    db_merged.save_local("faiss_index")


def load_db():
    db = FAISS.load_local("faiss_index", embedding_function,
                          allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    return retriever


# Start Flask server with CORS enabled
app = Flask(__name__)
CORS(app)


@app.route('/prompt', methods=['POST'])
def prompt():
    # Get prompt from request data
    prompt = request.json['name']
    print(prompt)
    # Get retriever
    retriever = load_db()
    model = Ollama(base_url='http://119.74.32.2:11434',
                   model="llama2", temperature=0.8)

    qachain = RetrievalQA.from_chain_type(model, retriever=retriever)
    output = qachain.invoke({"query": prompt})
    # Get response
    # response = retriever.get_relevant_documents(prompt)
    # print(response[0])
    # template = """Answer the question based only on the following context, which can include text:
    # {context}
    # Question: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)

    # model = ChatOpenAI(temperature=0, model="gpt-4-turbo")

    # chain = (
    #     {"context": retriever,
    #         "question": RunnablePassthrough()}
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )

    # output = chain.invoke(prompt)

    print(output)

    return output


if __name__ == "__main__":
    load_db()
    app.run(port=5000)
