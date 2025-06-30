import streamlit as st
import pandas as pd
import sqlite3
import os
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.callbacks import BaseCallbackHandler 


from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    query_intent: str = Field(description="A concise description of the user's intent in the natural language query.")
    target_tables: List[str] = Field(description="List of database tables likely involved in fulfilling the query.")
    expected_output_type: str = Field(description="Expected format of the data (e.g., 'table', 'single value', 'grouped aggregates').")
    key_metrics: List[str] = Field(default_factory=list, description="Key metrics or calculations requested (e.g., 'count', 'sum', 'average').")
    dimensions: List[str] = Field(default_factory=list, description="Dimensions or categories mentioned in the query.")


# --- Configuration ---
REGION = "us-central1" 
DB_PATH = "chinook.db"

# --- Database Setup ---
if not os.path.exists(DB_PATH):
    st.error(f"Database file '{DB_PATH}' not found.")
    st.info("Please make sure your 'chinook.db' file is in the same directory as this Streamlit app.")
    st.stop()

@st.cache_data
def get_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema_info = {}
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            column_names_types = [f"{col[1]} ({col[2]})" for col in columns_info]
            schema_info[table_name] = column_names_types
    except Exception as e:
        st.error(f"Error reading database schema: {e}")
        st.stop()
    finally:
        conn.close()
    return schema_info


class SQLCaptureCallbackHandler(BaseCallbackHandler):
    """Callback Handler to capture the executed SQL query."""
    def __init__(self):
        self.sql_queries = []

    def on_tool_start(self, tool: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""
        
        if tool['name'] == "sql_db_query":
            self.sql_queries.append(input_str)

    def on_tool_end(self, tool_output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        pass 

    def get_last_sql_query(self) -> Optional[str]:
        return self.sql_queries[-1] if self.sql_queries else None


@st.cache_resource
def get_langchain_components():
    
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-002",
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=REGION,
        temperature=0.1 
    )
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=db_toolkit,
        agent_type="openai-tools",
        verbose=False, 
        agent_kwargs={"max_iterations": 10},
    )
    return llm, db, sql_agent

llm, db, sql_agent = get_langchain_components()


# --- Streamlit UI ---
st.set_page_config(page_title="Visualize your data", layout="wide")
st.title("📊 SQL Agent ")
st.write("Enter your query!")



user_query = st.text_input("Enter your query here:", placeholder="e.g., Show me no. of invoices country wise")

if st.button("Generate & Visualize"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        st.subheader("Processing your request...")


        with st.spinner("Analyzing query intent..."):
            try:
                analysis_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert database query analyst. Extract precise structured details about the user's natural language query."),
                    ("user", "Analyze the following query and provide its structured details according to the 'QueryAnalysis' schema. If a field is not applicable, leave its list empty."),
                    ("user", "User Query: {query}"),
                ])

                analysis_llm = ChatVertexAI(
                    model_name="gemini-1.5-pro-002", 
                    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                    location=REGION,
                    temperature=0.1
                )
                query_analysis_chain = analysis_prompt | analysis_llm.with_structured_output(QueryAnalysis)
                query_metadata = query_analysis_chain.invoke({"query": user_query})

                

            except Exception as e:
                st.error(f"Error analyzing query metadata: {e}")
                st.info("Could not generate JSON metadata for the query. Proceeding anyway.")



        
        sql_callback_handler = SQLCaptureCallbackHandler()
        

        with st.spinner("Agent thinking and querying database..."):
            try:

                final_agent_output = sql_agent.invoke(
                    {"input": user_query},
                    config={"callbacks": [sql_callback_handler]}
                )

                executed_sql = sql_callback_handler.get_last_sql_query()

                st.subheader("Final Answer from Agent:")
                st.success(final_agent_output.get('output', 'No specific output from agent.')) 

                if executed_sql:
                    
                    st.subheader("SQL Query Executed by Agent:")
                    st.code(executed_sql, language="sql")
                    
                    
                    executed_sql_dic = eval(executed_sql)
                    executed_sql_query = executed_sql_dic['query']
                    

                    st.subheader("Extracted Data:")
                    
                    conn = sqlite3.connect(DB_PATH)
                    df_results = pd.read_sql_query(executed_sql_query, conn) 
                    conn.close()

                    if not df_results.empty:
                        st.dataframe(df_results)

                        st.subheader("Data Visualization:")

                        num_cols = df_results.select_dtypes(include=['number']).columns
                        obj_cols = df_results.select_dtypes(include=['object', 'datetime']).columns

                        if len(df_results) > 1 and not df_results.empty:
                            if len(obj_cols) >= 1 and len(num_cols) == 1:
                                try:
                                    st.bar_chart(df_results.set_index(obj_cols[0])[num_cols[0]])
                                    st.info(f"Showing a bar chart of '{num_cols[0]}' by '{obj_cols[0]}'.")
                                except Exception as chart_err: 
                                    st.info(f"Could not create bar chart for combination. Displaying raw data only. Error: {chart_err}")
                            elif len(num_cols) >= 2:
                                try:
                                    if len(df_results) > 1:
                                        st.line_chart(df_results, x=num_cols[0], y=num_cols[1])
                                        st.info(f"Showing a line chart of '{num_cols[1]}' vs. '{num_cols[0]}'.")
                                    else:
                                        st.info("Not enough data points for line chart. Displaying table only.")
                                except Exception as chart_err:
                                    st.info(f"Could not create line chart. Displaying raw data only. Error: {chart_err}")
                            elif len(num_cols) > 0 and len(obj_cols) == 0:
                                st.bar_chart(df_results[num_cols])
                                st.info(f"Showing a bar chart of numerical columns.")
                            else:
                                st.info("Automatic charting could not determine a suitable chart type. Displaying table only.")
                        else:
                            st.info("Not enough data rows or suitable columns to create a meaningful chart. Displaying table only.")
                    else:
                        st.warning("No data found for the given query.")
                else:
                    st.warning("Agent did not execute a SQL query for this request or an error occurred during SQL capture.")

            except Exception as e:
                st.error(f"Error during agent execution: {e}")
                st.info("The agent might have struggled to find an answer or generated an unrecoverable error. Check your query or the database schema.") # More helpful message

st.markdown("---")
st.caption("Created by Shivendra Sisodiya")
