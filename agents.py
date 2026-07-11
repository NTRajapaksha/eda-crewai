from crewai import Agent
from tools import DatasetProfileTool, StatisticalSummaryTool, OutlierDetectionTool
import os
import litellm

# --- GROQ FREE TIER COMPATIBILITY HACKS ---
# If you are using Groq, uncomment the lines below. Groq's API strictly rejects certain 
# kwargs (like cache_breakpoint) that CrewAI sends by default. This forces litellm to drop them.
# litellm.drop_params = True
# import crewai.llms.cache as _crewai_cache
# _crewai_cache.mark_cache_breakpoint = lambda msg: msg
# ------------------------------------------

from dotenv import load_dotenv
load_dotenv()

# For premium/paid usage (e.g. OpenAI), set this to your desired model.
# Ensure you have OPENAI_API_KEY in your .env file.
# If using Groq, change this to "groq/llama-3.3-70b-versatile" and uncomment the hacks above.
DEFAULT_LLM = "ollama/phi3:mini"

ingestion_agent = Agent(
    role='Data Ingestion Specialist',
    goal='Load the dataset, understand its structure, and produce a clear profile that other agents can reason from',
    backstory='An expert in first-contact data assessment who has seen thousands of messy CSVs and knows how to quickly surface the most important structural facts about a new dataset',
    verbose=True,
    cache=False,
    allow_delegation=False,
    max_iter=3,
    tools=[DatasetProfileTool()],
    llm=DEFAULT_LLM
)

statistical_agent = Agent(
    role='Statistical Analyst',
    goal='Compute comprehensive descriptive statistics across all columns and identify the most statistically significant patterns and relationships in the data',
    backstory='A quantitative analyst who thinks in distributions, correlations, and percentiles, and who never interprets data without first computing the numbers',
    verbose=True,
    cache=False,
    allow_delegation=False,
    max_iter=3,
    tools=[StatisticalSummaryTool()],
    llm=DEFAULT_LLM
)

interpretation_agent = Agent(
    role='Data Storyteller',
    goal='Translate the statistical outputs into a clear, plain-English narrative that a non-technical stakeholder could understand — what the data looks like, what stands out, what is concerning',
    backstory='A business intelligence communicator who bridges the gap between raw statistics and human-readable insight, specialising in making data findings accessible without losing accuracy',
    verbose=True,
    cache=False,
    allow_delegation=False,
    max_iter=3,
    tools=[], # No custom tools, reasons from previous outputs
    llm=DEFAULT_LLM
)

anomaly_agent = Agent(
    role='Data Quality Inspector',
    goal='Identify every data quality red flag in the dataset — impossible values, placeholder strings, outliers, inconsistencies — and produce a clear actionable list of issues for a data engineer to fix',
    backstory='A detail-obsessed data quality engineer who has debugged production pipelines and knows that silent data errors cause the most damage — the kind of person who checks every assumption',
    verbose=True,
    cache=False,
    allow_delegation=False,
    max_iter=3,
    tools=[OutlierDetectionTool()],
    llm=DEFAULT_LLM
)

report_agent = Agent(
    role='Technical Report Writer',
    goal='Assemble the outputs of all previous agents into a single, coherent, well-structured EDA report with clear sections, and write it to disk',
    backstory='A technical writer with a data science background who knows how to organise complex findings into a report that is both rigorous and readable — the kind of document you would share with a client or senior stakeholder',
    verbose=True,
    cache=False,
    allow_delegation=False,
    max_iter=3,
    tools=[], # Report will be saved natively by Python in main.py
    llm=DEFAULT_LLM
)
