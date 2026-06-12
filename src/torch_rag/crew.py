from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from torch_rag.tools.qdrant_search import QdrantSearchTool
import os
from dotenv import load_dotenv

load_dotenv()

# initialize tools
VectorSearchTool = QdrantSearchTool()

# initialize LLMs
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name1 = os.getenv("MODEL1")
model_name2 = os.getenv("MODEL2")
api_base = os.getenv("API_BASE")

llm = LLM(model=model_name1, api_base=api_base)
llm2 = LLM(model=model_name2, api_base=api_base)
gpt5_llm = LLM(model="openai/gpt-5.4-mini", api_key=openai_api_key)

@CrewBase
class TorchRag():
    """TorchRag crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['data_retrieval_agent'], # type: ignore[index]
            tools = [VectorSearchTool],
            llm = llm2,
            allow_delegation=False,
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer_agent'], # type: ignore[index]
            llm = llm2,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            agent=self.researcher()
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            agent=self.reporting_analyst(),
            context=[self.research_task()],
            output_file='output/QnA_report.json'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TorchRag crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
