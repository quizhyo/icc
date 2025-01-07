from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
from typing import Optional, Any

class DataAnalysisAgent:
    def __init__(
        self,
        df: pd.DataFrame,
        llm: ChatOpenAI,
        agent_type: AgentType,
        verbose: bool = False,
        return_intermediate_steps: bool = False
    ):
        self.df = df
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps

    def create_agent(self):
        """Create and return a pandas DataFrame agent"""
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            agent_type=self.agent_type,
            verbose=self.verbose,
            return_intermediate_steps=self.return_intermediate_steps,
            allow_dangerous_code=True
        )

    def analyze(self, query: str) -> dict:
        """Run analysis on the data"""
        agent = self.create_agent()
        result = agent.invoke({"input": query})
        return result