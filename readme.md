# Running a LLM with langchain and FastAPI


Step 1: Setting Up the Environment
First, install the necessary dependencies.

pip install openai langchain chromadb fastapi uvicorn

Step2: Start the API server with:

uvicorn main:app --reload
Then, test the endpoint:

curl "http://127.0.0.1:8000/ask?question=What is an AI agent?"
