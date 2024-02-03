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

        instructions = """Run AI predictions on an agent.
        Use a single neural network and perform calculations
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