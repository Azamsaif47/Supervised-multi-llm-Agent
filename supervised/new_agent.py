import os
import openai
from langchain_community.utilities import SQLDatabase
import pymysql
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
import ast
import re
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema import SystemMessage

# Set the environment variable for your OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-proj-zgXqKiXJ3flv7xFoBvdjT3BlbkFJ1hezS1kRuOsw0jTkBEjE"
openai.api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is loaded correctly
if openai.api_key is None:
    raise ValueError("OpenAI API key is not found. Please ensure it is set as an environment variable.")

# Define db as a global variable
db = None

def get_db():
    global db  # Declare that we are using the global db variable
    
    if db is None:  # Initialize db only if it's not already set
        # Path to the CA.pem file
        ca_file_path = "cassle.pem"

        # MySQL credentials and database details
        db_uri = "mysql+pymysql://avnadmin:AVNS_RqUZqAf0WXtfB_iIsot@employees-emplo.i.aivencloud.com:16429/employees?ssl_ca=" + ca_file_path

        # Create and set the global db instance
        db = SQLDatabase.from_uri(db_uri)
    
    return db

db = get_db()

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

def clean_query(query: str) -> str:
    """Remove any unwanted prefixes or formats from the query string."""
    return query.replace('SQLQuery: ', '')

# Initialize the language model and SQL query chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = create_sql_query_chain(llm, db)

# Example usage
response = chain.invoke({"question": "How many employees are there"})
print("Generated Query:", response)

# Clean the query if needed
cleaned_query = clean_query(response)
print("Cleaned Query:", cleaned_query)

# Execute the cleaned query
try:
    result = db.run(cleaned_query)
    print("Query Result:", result)
except Exception as e:
    print("Error executing query:", e)



chain.get_prompts()[0].pretty_print()

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": "How many employees have more than 80000 salary"})

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

chain.invoke({"question": "How many senior engineers are there"})

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

examples = [
    {"input": "List all employees.", "query": "SELECT * FROM employees;"},
    {
        "input": "Find all employees in the 'Development' department.",
        "query": "SELECT * FROM employees WHERE emp_no IN (SELECT emp_no FROM dept_emp WHERE dept_no = (SELECT dept_no FROM departments WHERE dept_name = 'Development'));",
    },
    {
        "input": "List all employees with the title 'Senior Engineer'.",
        "query": "SELECT * FROM employees WHERE emp_no IN (SELECT emp_no FROM titles WHERE title = 'Senior Engineer');",
    },
    {
        "input": "Find the total salaries of all employees.",
        "query": "SELECT SUM(salary) FROM salaries;",
    },
    {
        "input": "List all employees hired in the year 2020.",
        "query": "SELECT * FROM employees WHERE YEAR(hire_date) = 2020;",
    },
    {
        "input": "How many employees are in the department with ID 5?",
        "query": "SELECT COUNT(*) FROM dept_emp WHERE dept_no = 5;",
    },
    {
        "input": "Find the total number of departments.",
        "query": "SELECT COUNT(*) FROM departments;",
    },
    {
        "input": "List all employees with a salary greater than $50,000.",
        "query": "SELECT * FROM employees WHERE emp_no IN (SELECT emp_no FROM salaries WHERE salary > 50000);",
    },
    {
        "input": "Who are the top 5 highest paid employees?",
        "query": "SELECT emp_no, SUM(salary) AS TotalSalary FROM salaries GROUP BY emp_no ORDER BY TotalSalary DESC LIMIT 5;",
    },
    {
        "input": "Which employees were hired in the year 2000?",
        "query": "SELECT * FROM employees WHERE YEAR(hire_date) = 2000;",
    },
    {
        "input": "How many employees are there?",
        "query": "SELECT COUNT(*) FROM employees;",
    },
]


from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)


from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

One more thing if someone ask you a simple question like "hi" , "how are you " and something like that give answer from your pre-trained llm model in such cases

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

And one more important thing to consider in your output as i am using voice assistant for your output so everything you output
the assistant will say it so while creating a list of employees are something like that take care that your output is suitable to say
i mean if you print numbers in front of employees it will also say that and also name the employees in such a way that the voice  
assistant does not say it rapidly i want some gap in saying one name then another

Also remove the curly brackets from output while printing the results 

If the question does not seem related to the database, just return "I don't know" as the answer.
 
Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
)





def query_agent(agent, question):
    try:
        response = agent.invoke(question,return_only_outputs=True,)
        return response
    except Exception as e:
        print(f"Error querying agent: {e}")
        return {"response": "Error querying agent"}

question = "How many employees are in the database"  # Replace with your question
response = agent.invoke(question)
print(f"Response: {response}")