from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

def doc_from_pc(path, doc):
    loader = DirectoryLoader(path, doc)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    split_document = splitter.split_documents(docs)
    return split_document

def create_database(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-3.5-turbo", #You can choose between different OpenAI models. gpt.3.5 is sufficient for my purposes.
        temperature=0.4 #Defines the randomness/creativity of the response. Closer to 0 is less creative and closer to 1 is more creative.
    )
    #Modify the prompt below for your purpose
    prompt = ChatPromptTemplate.from_template("""
    You are a customer service assistant. Answer the customer service rep's question: 
    Question: {input}
    Context: {context}                                                                                        
    """)
    chain = create_stuff_documents_chain(
        llm = model,
        prompt = prompt
    )
    retriever = vectorStore.as_retriever(search_kwargs={"k":4})
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )
    return retrieval_chain

def chatting(chain, user_input):
    response = chain.invoke({
        "input": user_input,
    })
    return (response["answer"])

if __name__ == "__main__":
    docs = doc_from_pc(r'C:\Users\YOURUSER\Desktop\personalchatbot', "trainingdoc.txt") #replace YOURUSER with your usernmame. You can also copy the complete path of the folder where your document is located and replace the entire string.
    vectorStore = create_database(docs)
    chain = create_chain(vectorStore)

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "break":
            break

        response = chatting(chain, user_input)
        print("Assistant:", response)
