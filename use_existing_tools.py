from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

Settings.llm= OpenAI(model="gpt-4o", temperature=0)

#function tools
def multiply(b:float, c:float) -> float:
    """Mulitply two numbers and return product"""
    return b * c

multiply_tool=FunctionTool.from_defaults(fn=multiply)

def add(b:float, c:float) -> float:
    """Add two number and return sum"""
    return b + c

add_tool=FunctionTool.from_defaults(fn=add)

finance_tools=YahooFinanceToolSpec().to_tool_list()

finance_tools.extend([multiply_tool, add_tool])

agent=ReActAgent.from_tools(finance_tools, verbose=True)

response=agent.chat("What is the current price of NVDA?")
print(response)

response2=agent.chat("what is the current price of adani")
print(response2)