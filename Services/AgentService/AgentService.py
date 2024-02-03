from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool

from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI

from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatAnthropic

class AgentService:
    def __init__(self) -> None:
        tools = [PythonREPLTool()]

        instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt = base_prompt.partial(instructions=instructions)

        agent = create_openai_functions_agent(ChatOpenAI(temperature=0), tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    def predict(self, input_data):
        prediction_data = self.agent_executor.invoke(
            {
                "input": """Understand, write a single neuron neural network in PyTorch.
                Generate data from"""
            }
        )
        return prediction_data