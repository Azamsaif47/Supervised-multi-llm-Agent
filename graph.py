from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
import functools
import operator
from typing import Sequence, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Annotated
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from pdf_agent import pdf_agent
from sql_agent import agent

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

'''
def agent_node(state, agent, name):
    messages = state['messages']
    combined_messages = " ".join([msg.content for msg in messages])
    result = agent.invoke({"input": combined_messages})
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
'''
def agent_node(state, agent, name):
    messages = state['messages']
    combined_messages = " ".join([msg.content for msg in messages])

    if name == "pdf_agent":
        result = agent.invoke({"query": combined_messages})
        output_content = result.get("result", "")
    else:
        result = agent.invoke({"input": combined_messages})
        output_content = result.get("output", "")

    return {"messages": [HumanMessage(content=output_content, name=name)]}



members = ["sql_agent", "pdf_agent"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4o-mini")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

sql_agent = agent
sql_node = functools.partial(agent_node, agent=sql_agent, name="sql_agent")

pdf_agent = pdf_agent
pdf_node = functools.partial(agent_node, agent=pdf_agent, name="pdf_agent")

workflow = StateGraph(AgentState)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("pdf_agent", pdf_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

def process_input(input_message: str) -> str:
    initial_state = {
        "messages": [
            HumanMessage(content=input_message)
        ]
    }

    output_content = None
    for state in graph.stream(initial_state):
        print(f"Intermediate state: {state}")  # Debug statement to print intermediate states
        # Check if the state corresponds to a specific agent
        for agent in members:
            if agent in state and 'messages' in state[agent]:
                # Extract the content from the last message
                output_content = state[agent]['messages'][-1].content
        
        # Check if the supervisor state is in the result and if it indicates FINISH
        if 'supervisor' in state and state['supervisor'].get('next') == 'FINISH':
            break

    if output_content:
        return output_content


