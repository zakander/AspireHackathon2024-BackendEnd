from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool

from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI

from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatAnthropic

class AgentService:
    def __init__(self) -> None:
        # Set up agent executor
        tools = [PythonREPLTool()]

        instructions = """Run AI predictions on an agent.
        Use a single neural network and perform calculations
        """
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt = base_prompt.partial(instructions=instructions)

        agent = create_openai_functions_agent(ChatOpenAI(temperature=0), tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    def predict(self, input_data):
        # Prompt is constructed from sample input data, and is used to make predictions based on candidate's age, capability and associate level
        invoke_prompt = "List some recommmend courses for a " + str(input_data["age"]) + " year old " + input_data["level"] + " associate who has some basic " + input_data["capability"] + " qualifications"
        prediction_data = self.agent_executor.invoke(
            {
                "input": invoke_prompt,
            }
        )
        return prediction_data