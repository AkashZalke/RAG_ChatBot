from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
os.environ["OPENAI_API_KEY"]="your_key"

def main():
    os.environ["OPENAI_API_KEY"]="your_key"

    loader = PyPDFLoader("pdf/Lic-brochure-917-Single-Endoment-Plan-2021.pdf")
    docs = loader.load()
    # print(docs[1])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    url = "https://6bc84abd-39b5-4c2a-8253-846d0f68af30.europe-west3-0.gcp.cloud.qdrant.io"
    api_key = "tjrxDYxaul4rnd-XpZvHoWL0GPyw5Kp8xw_IJp87pLBqjz0r9Rw8gQ"

    qdrant = Qdrant.from_documents(
        documents=all_splits,
        embedding=embeddings,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name="my_documents",
    )
    query = "What are Death Benefits of LIC Single Premium Endowment Plan ?"

    found_docs = qdrant.similarity_search_with_score(query)
    # retriever= qdrant.as_retriever()
    context=''

    for k in found_docs:
        context+=k[0].page_content

    llm = ChatOpenAI()

    sys_prompt: PromptTemplate = PromptTemplate(
        template=f"""You are an LIC agent
        
        The context in as follows
        {context}
        
        """
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    student_prompt: PromptTemplate = PromptTemplate(
        template=query
    )
    student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt,student_message_prompt])


    rag_chain = (
        chat_prompt
        | llm
        | StrOutputParser()
    )
    d={'input':query}
    res=rag_chain.invoke(d)
    
    

main()
