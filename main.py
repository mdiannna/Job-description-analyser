from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request, HTTPException


import uvicorn
import getpass

import os
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


# embeddings = OpenAIEmbeddings()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

texts = ["AI agents are autonomous decision-making systems.", "Vector databases help store and retrieve embeddings efficiently."]
vector_store.add_texts(texts)



prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an AI agent that provides answers based on knowledge retrieval.
    Question: {question}
    Answer:
    """
)


# llm = OpenAI()
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

chain = LLMChain(llm=llm, prompt=prompt)

app = FastAPI()
templates = Jinja2Templates("templates")

app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")
async def read_root(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})


@app.get("/ask")
def ask_agent(question: str):
    relevant_docs = vector_store.similarity_search(question, k=1)
    context = " ".join([doc.page_content for doc in relevant_docs])
    response = chain.run(question=question + " " + context)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)