from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def init_llm_langchain(llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an AI agent that provides answers based on knowledge retrieval.
        Question: {question}
        Answer:
        """
    )


    chain = LLMChain(llm=llm, prompt=prompt)
    return chain