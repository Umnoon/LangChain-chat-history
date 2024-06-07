from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

documentation_template = """As a documentation bot, your goal is to provide accurate and helpful information about
     Sagemaker. You should answer user inquiries based on the context provided. If he greets, then greet him. Don't include prefix 'Answer'.
     
     Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
 <ctx>
     {context} 
    </ctx>
<hs> {history} </hs>
    Question: {question}"""

DOCUMENT_PROMPT = PromptTemplate(
    template=documentation_template, input_variables=["history", "context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    chain_type_kwargs={
        "verbose": True,
        "prompt": DOCUMENT_PROMPT,
        "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
    },
    verbose=True,
)
