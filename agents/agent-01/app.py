from dynamiq.nodes.llms.groq import Groq
from dynamiq.connections import Groq as GroqConnection, Tavily as TavilyConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.tavily import TavilyTool

import os
from dotenv import load_dotenv

load_dotenv()


# Setup your Tavily tool
tavily_tool = TavilyTool(
    connection=TavilyConnection(api_key=os.getenv("TAVILY_API_KEY")),
)

# Setup your LLM
llm = Groq(
    id="groq",
    connection=GroqConnection(api_key=os.getenv("GROQ_API_KEY")),
    model="mixtral-8x7b-32768",
    temperature=0.3,
    max_tokens=1000,
)

# Create the ReAct agent
agent = ReActAgent(
    name="react-agent",
    llm=llm, # Language model instance
    tools=[tavily_tool],  # List of tools that the agent can use
    role="Search Analyst",  # Role of the agent
    max_loops=10, # Limit on the number of processing loops
    
)

# Run the agent with an input
result = agent.run(
    input_data={
        "input": "Champions Trophy 2025 Final Match, Which teams are playing the final match?",
    }
)

print(result.output.get("content"))