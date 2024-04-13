import glob
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from llm import load_and_split_pdf, generate_summary
from image import generate_img_summaries

pdf_files = glob.glob('/Users/aaron/Documents/**/*.pdf', recursive=True)
# print(pdf_files)

img_files = glob.glob('/Users/aaron/Documents/**/*.jpg', recursive=True)
print(img_files)


def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        print(doc_summaries)
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


# The vectorstore to use to index the summaries
vectorstore = Chroma(
    collection_name="fh_local_rag", embedding_function=OpenAIEmbeddings()
)

pages = load_and_split_pdf("./style.pdf")
texts = [page.page_content for page in pages]
text_summaries = generate_summary(texts)

img_base64_list, image_summaries = generate_img_summaries(
    "/Users/aaron/Documents")

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    image_summaries,
    img_base64_list,
)

# Prompt template
template = """Answer the question based only on the following context, which can include text:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Option 1: LLM
model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
# Option 2: Multi-modal LLM
# model = Ollama(base_url='http://119.74.32.2:11434',
#                model="llava", temperature=0)

# RAG pipeline
chain = (
    {"context": retriever_multi_vector_img, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke(
    "I think there is an image somewhere of an Asian male. Can you describe him? How would you dress him using the clothes from Style Inc?"
)

print(output)

# folder_hierarchy = get_folder_hierarchy()
# print(folder_hierarchy)
