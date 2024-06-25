from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response= query_engine.query(
    "What is the total amount of the Canadian Federal budget"
)
print(response)