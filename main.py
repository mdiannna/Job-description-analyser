# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request, HTTPException

from llm_init import init_llm_and_embeddings
from langchain_init import init_llm_langchain
import uvicorn


llm, embeddings = init_llm_and_embeddings()
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

texts = ["AI agents are autonomous decision-making systems.", "Vector databases help store and retrieve embeddings efficiently."]
vector_store.add_texts(texts)


chain = init_llm_langchain(llm)

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