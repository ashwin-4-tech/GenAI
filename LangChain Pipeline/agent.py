import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, load_tools

load_dotenv()

def run_agent():
    """
    Demonstrates an Agent that uses Wikipedia to find real-time information.
    """
    print("--- Starting Intelligent Agent ---")

    # 1. Initialize the LLM
    # We use temperature=0 because we want facts, not creativity, when searching.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 2. Load Tools
    # These are the "hands" of the agent. We are giving it access to Wikipedia.
    tools = load_tools(["wikipedia"])

    # 3. Get the standard Agent Prompt from the LangChain Hub
    # The Hub is a community repository of pre-written, highly effective prompts.
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 4. Construct the Agent
    # The agent is the "brain" that figures out WHICH tool to use.
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    # The executor is the "engine" that actually runs the agent in a loop until it gets an answer.
    # verbose=True is critical here so beginners can see it deciding to search Wikipedia.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 6. Give the agent a mission
    # The LLM doesn't natively know recent hyper-specific trivia, so it must search.
    agent_executor.invoke({"input": "What is LangChain and what year was it launched?"})

if __name__ == "__main__":
    run_agent()