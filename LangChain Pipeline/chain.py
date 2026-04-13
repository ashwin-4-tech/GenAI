import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (like your OPENAI_API_KEY) from a .env file
load_dotenv()

def run_basic_chain():
    """
    Demonstrates how to connect a Prompt Template to an LLM.
    """
    print("--- Starting Basic Chain ---")

    # 1. Initialize the LLM
    # 'temperature' controls creativity: 0 is deterministic/robotic, 1 is highly creative.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 2. Create the Prompt Template
    # We use {topic} as a placeholder variable that we will fill in later.
    template = "You are a technical expert. Explain {topic} to a 10-year-old in two sentences."
    prompt = PromptTemplate.from_template(template)

    # 3. Build the Chain using LCEL (LangChain Expression Language)
    # The '|' operator magically connects the prompt output directly into the LLM input.
    chain = prompt | llm

    # 4. Invoke (run) the chain with our specific data
    result = chain.invoke({"topic": "Cloud Computing"})
    
    print(f"Result:\n{result.content}\n")

if __name__ == "__main__":
    run_basic_chain()