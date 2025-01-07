from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

class LegalAgentTeam:
    def __init__(self, knowledge_base: PDFKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.legal_researcher = self._create_researcher()
        self.contract_analyst = self._create_analyst() 
        self.legal_strategist = self._create_strategist()
        self.team_lead = self._create_team_lead()

    def _create_researcher(self):
        """Create Legal Researcher agent"""
        return Agent(
            name="Legal Researcher",
            role="Legal research specialist",
            model=OpenAIChat(model="gpt-4"),
            tools=[DuckDuckGo()],
            knowledge=self.knowledge_base,
            search_knowledge=True,
            instructions=[
                "Find and cite relevant legal cases and precedents",
                "Provide detailed research summaries with sources",
                "Reference specific sections from the uploaded document",
                "Always search the knowledge base for relevant information"
            ],
            show_tool_calls=True,
            markdown=True
        )

    def _create_analyst(self):
        """Create Contract Analyst agent"""
        return Agent(
            name="Contract Analyst",
            role="Contract analysis specialist",
            model=OpenAIChat(model="gpt-4"),
            knowledge=self.knowledge_base,
            search_knowledge=True,
            instructions=[
                "Review contracts thoroughly",
                "Identify key terms and potential issues",
                "Reference specific clauses from the document"
            ],
            markdown=True
        )

    def _create_strategist(self):
        """Create Legal Strategist agent"""
        return Agent(
            name="Legal Strategist",
            role="Legal strategy specialist",
            model=OpenAIChat(model="gpt-4"),
            knowledge=self.knowledge_base,
            search_knowledge=True,
            instructions=[
                "Develop comprehensive legal strategies",
                "Provide actionable recommendations",
                "Consider both risks and opportunities"
            ],
            markdown=True
        )

    def _create_team_lead(self):
        """Create Team Lead agent"""
        return Agent(
            name="Legal Team Lead",
            role="Legal team coordinator",
            model=OpenAIChat(model="gpt-4"),
            team=[self.legal_researcher, self.contract_analyst, self.legal_strategist],
            knowledge=self.knowledge_base,
            search_knowledge=True,
            instructions=[
                "Coordinate analysis between team members",
                "Provide comprehensive responses",
                "Ensure all recommendations are properly sourced",
                "Reference specific parts of the uploaded document",
                "Always search the knowledge base before delegating tasks"
            ],
            show_tool_calls=True,
            markdown=True
        )

    def analyze(self, query: str, analysis_type: str):
        """Run analysis based on analysis type"""
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
            },
            "Custom Query": {
                "query": query,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
            }
        }

        if analysis_type in analysis_configs:
            config = analysis_configs[analysis_type]
            formatted_query = f"""
            Using the uploaded document as reference:
            
            {config['query'] if analysis_type != 'Custom Query' else query}
            Focus Areas: {', '.join(config['agents'])}
            
            Please search the knowledge base and provide specific references from the document.
            """
            return self.team_lead.run(formatted_query)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")