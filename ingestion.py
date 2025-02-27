from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

urls    = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

loaded_docs     = [WebBaseLoader(url).load() for url in urls]
documents_list  = [item for doc in loaded_docs for item in doc]

# Split documents_list into chunks
text_splitter_obj   = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=50)
documents_split     = text_splitter_obj.split_documents(documents_list)

# chroma_vectorstore  = Chroma.from_documents(
#     documents=documents_split,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma"
# )

chroma_retriever    = Chroma(
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./.chroma"

).as_retriever()
print(documents_split)