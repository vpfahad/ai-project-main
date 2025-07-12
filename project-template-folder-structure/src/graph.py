from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from typing import Annotated, List
from .tools import get_available_tools
from langchain.prompts import ChatPromptTemplate
from src.agentwrap.easy_llm import load_chat_model
from datetime import datetime


# System and summary prompts
SQL_AGENT_SYSTEM_PROMPT = f"""
        You are QueryAgent, a specialized agent designed to assist users in managing and querying a database.
        Your purpose is to transform natuarl language requests into SQL queries, execute them, and return the results in a user-friendly format
        <instructions>
        1. **Understand User Intent**: Analyze the user's request to determine the underlying intent and required data. 
        2. **Use Available Tools**: If the request requires information about the database schema, use the `list_tables`, `describe_table`, or `sample_table` tools to gather necessary details.
        3. **Generate SQL Queries**: Based on the user's request and the gathered information, construct appropriate SQL queries.
        4. **Execute SQL Queries**: Use the `execute_sql` tool to run the generated SQL queries against the database.
        5. **Return Results**: Return only the markdown version of the data extracted from the table.
        </instructions>

        
        ## DB SCHEMA

        The database has the following tables on the schema `onlysql`. You should only access the tables on this schema.

        [creators]
        id: int8 (Primary key)
        first_name: text
        last_name: text
        email: text
        join_date: timestamptz
        last_post_date: timestamptz

        [customers]
        id: int8 (Primary key)
        first_name: text
        last_name: text
        email: text
        join_date: timestamptz

        [transactions]
        id: int8 (Primary key)
        customer_id: int8 (Foreign key to customers.id)
        creator_id: int8 (Foreign key to creators.id)
        transaction_date: timestamptz
        amount_usd: float8
        transaction_type: text


        Today's date is {datetime.now().strftime('%Y-%m-%d')}.

        Your responses should be clear, concise, and directly address the user's request. If you need to use a tool, provide a brief explanation of why the tool is necessary before invoking it.
        If you encounter any issues or need clarification, ask the user for more information.

        """
# Fix SUMMARY_PROMPT to use a format string, not an f-string with undefined variable
SUMMARY_PROMPT = """
Summarize the following table :{sql_results}
Result format:

Table :<<show the table if any>>

Summary: <<summary on the table>>
"""

class SQLAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []
    sql_results: str = ""
    summary: str = ""

class SQLQueryAgent:
    def __init__(self, model_name="google_genai", temperature=0.1):
        self.llm = load_chat_model(model_name=model_name, temperature=temperature).bind_tools(get_available_tools())
        self.summary_llm = load_chat_model(model_name=model_name, temperature=0.1)
        self.system_prompt = SQL_AGENT_SYSTEM_PROMPT
        self.summary_prompt = SUMMARY_PROMPT
        self.runnable = self.build_graph()

    #FIXME: Need proper structed oupt from the response to go the summary node
    def build_graph(self):
        def agent_node(state: SQLAgentState) -> SQLAgentState:
            response = self.llm.invoke(
                [SystemMessage(content=self.system_prompt)] + state.messages
            )
            state.messages = state.messages + [response]
            # If the response contains SQL results, store them
            if hasattr(response, "content"):
                state.sql_results = response.content
            return state

        def summary_node(state: SQLAgentState) -> SQLAgentState:
            # Generate a summary from the last message instead of sql_results
            last_message_content = state.messages[-1].content if state.messages else ""
            prompt = ChatPromptTemplate.from_template(self.summary_prompt).format(sql_results=last_message_content)
            summary_response = self.summary_llm.invoke([HumanMessage(content=prompt)])
            state.summary = summary_response.content
            state.messages.append(summary_response)
            return state

        def router(state: SQLAgentState) -> str:
            last_message = state.messages[-1]
            # If the last message is a tool call, go to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools_node"
            # If SQL results are present, go to summary
            if state.sql_results:
                return "summary_node"
            return END

        builder = StateGraph(SQLAgentState)
        builder.add_node("agent_node", agent_node)
        builder.add_node("tools_node", ToolNode(get_available_tools()))
        builder.add_node("summary_node", summary_node)

        builder.add_edge(START, "agent_node")
        builder.add_conditional_edges("agent_node", router, ["tools_node", "summary_node", END])
        builder.add_edge("tools_node", "agent_node")
        builder.add_edge("summary_node", END)

        return builder.compile(checkpointer=MemorySaver())

    def invoke(self, message: str) -> str:
        config = {
            "configurable": {
                "thread_id": "1"
            }
        }
        result = self.runnable.invoke(
            input={"messages": [HumanMessage(content=message)]},
            config=config,
        )
        # Return the summary if available, else the last message
        #print(result)
        #return result.get("summary") or result["messages"][-1].content
        return result.get("summary") 
# Example usage
if __name__ == "__main__":
    agent = SQLQueryAgent()
    user_query = "show me the first five records of customers table?"
    print("\n\n")
    print(agent.invoke(user_query))
