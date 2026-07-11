# Multi-Agent EDA Pipeline
### Framework: CrewAI | Models: Local Ollama | Dataset: Retail Store Sales Dirty

---

## What You Are Building
A team of 5 specialized CrewAI agents that collectively performs exploratory data analysis on the Retail Store Sales Dirty dataset, produces a written EDA report, and flags anomalies — fully autonomously. Each agent has a defined role, a goal, and a set of tools. They hand off work to each other in sequence, with one agent's output becoming the next agent's input.

---

## Overview
| | |
|---|---|
| **Framework** | CrewAI |
| **Models** | `llama3.2:3b` and `phi3:mini` via Ollama (local, free) |
| **Dataset** | Kaggle: Retail Store Sales — Dirty for Data Cleaning (12,575 rows, 8 columns) |
| **Agents** | Ingestion Agent, Statistical Agent, Interpretation Agent, Anomaly Agent, Report Agent |
| **Final output** | A structured plain-English EDA report per model + a comparison table across models |
| **Cost** | Completely free |

---

## Part 1 — Environment Setup

### Step 1 — Install Ollama and Pull Models
1. Go to ollama.com and download Ollama for your OS. The installer is a single file — run it and Ollama starts as a background service automatically.
2. Open a terminal and pull your two models:
   - `ollama pull llama3.2:3b`
   - `ollama pull phi3:mini`
3. Confirm both are available by running `ollama list` — you should see both models listed with their sizes.
4. Ollama runs a local server on port 11434 by default. Keep it running in the background while you work.

### Step 2 — Install CrewAI
1. Create a new Python virtual environment for this project — this keeps dependencies isolated. Name it something clear like `crewai-eda-env`.
2. Activate the virtual environment.
3. Install CrewAI: `pip install crewai crewai-tools`
4. Install supporting libraries: `pip install pandas numpy scipy ollama`
5. Verify CrewAI is installed correctly by running `crewai --version` in the terminal — you should see a version number.

### Step 3 — Connect CrewAI to Ollama
1. CrewAI uses LiteLLM under the hood to talk to models. To point it at your local Ollama instance, you configure the LLM using the `ollama/` prefix in the model name (e.g. `ollama/llama3.2:3b`).
2. Set the `OPENAI_API_KEY` environment variable to any placeholder string (e.g. `"NA"`) — CrewAI expects this variable to exist even when you are not using OpenAI. Without it, CrewAI will throw an error on startup.
3. Set the Ollama base URL environment variable: `OPENAI_API_BASE=http://localhost:11434` — this tells LiteLLM where your Ollama server is.
4. You do not need an actual OpenAI key. These two env vars are just scaffolding for CrewAI's LLM configuration layer.

### Step 4 — Download the Dataset
1. Go to `kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning`.
2. Download the CSV file and save it in your project folder as `retail_sales_dirty.csv`.
3. Do a quick manual inspection — open it in Excel or a text editor to see the columns, spot obvious issues (ERROR/UNKNOWN strings, missing values, mixed formats), and understand the data before the agents touch it.

### Step 5 — Project Folder Structure
Organise your project folder before writing any code:
```
project1_eda_pipeline/
├── retail_sales_dirty.csv       ← your dataset
├── main.py                      ← where you define and run the crew
├── agents.py                    ← agent definitions
├── tasks.py                     ← task definitions
├── tools.py                     ← custom tools your agents will use
└── outputs/                     ← where agent reports get saved
```
CrewAI separates agents (who they are), tasks (what they do), and tools (what they can call) into distinct concepts. Keeping them in separate files makes it much easier to debug and iterate.

---

## Part 2 — Understanding CrewAI Concepts

Before building, understand these four CrewAI building blocks:

**Agent**: a role with a goal and a backstory. The role defines what the agent specialises in, the goal defines what it is trying to achieve in a given run, and the backstory gives it context about why it exists. These are plain text strings — no code logic here.

**Task**: a specific piece of work assigned to an agent. A task has a description (what to do), an expected output (what format the result should be in), and is assigned to exactly one agent.

**Tool**: a Python function that an agent can call to interact with the real world — reading a file, computing statistics, writing a report. Tools are the only way agents take real actions beyond reasoning.

**Crew**: the orchestrator that holds a list of agents and tasks, defines the execution order (sequential or hierarchical), and runs the whole pipeline.

---

## Part 3 — Build the Custom Tools

Tools are what let your agents compute real statistics rather than hallucinate them. Build these before defining agents or tasks:

### Tool 1 — Dataset Profile Tool
This tool loads the CSV and returns a structured text summary: column names, dtypes as stored, number of unique values per column, percentage missing per column, and 5 sample rows formatted as a readable table. This is what the Ingestion Agent will call.

### Tool 2 — Statistical Summary Tool
This tool computes: mean, median, std, min, max, and 25th/75th percentiles for all numeric columns; value counts (top 5 most frequent values) for all categorical columns; a Pearson correlation matrix for numeric columns; and the class distribution of the target column (Payment Method). Returns all of this as structured text. The Statistical Agent uses this tool.

### Tool 3 — Outlier Detection Tool
This tool computes, per numeric column: the number of values beyond 3 standard deviations (Z-score method), the number of values beyond 1.5×IQR (IQR method), and any string values that appear to be placeholders (ERROR, UNKNOWN, N/A, -, 0 in a column where 0 is impossible). Returns a flagged list per column. The Anomaly Agent uses this tool.

### Tool 4 — Report Writer Tool
This tool takes a text string and writes it to a file in the outputs/ folder with a timestamped filename. The Report Agent uses this to persist the final EDA report to disk.

---

## Part 4 — Define the 5 Agents

### Agent 1 — Ingestion Agent
- **Role**: Data Ingestion Specialist
- **Goal**: Load the dataset, understand its structure, and produce a clear profile that other agents can reason from
- **Backstory**: An expert in first-contact data assessment who has seen thousands of messy CSVs and knows how to quickly surface the most important structural facts about a new dataset
- **Tools**: Dataset Profile Tool
- **LLM**: `ollama/llama3.2:3b`

### Agent 2 — Statistical Agent
- **Role**: Statistical Analyst
- **Goal**: Compute comprehensive descriptive statistics across all columns and identify the most statistically significant patterns and relationships in the data
- **Backstory**: A quantitative analyst who thinks in distributions, correlations, and percentiles, and who never interprets data without first computing the numbers
- **Tools**: Statistical Summary Tool
- **LLM**: `ollama/llama3.2:3b`

### Agent 3 — Interpretation Agent
- **Role**: Data Storyteller
- **Goal**: Translate the statistical outputs into a clear, plain-English narrative that a non-technical stakeholder could understand — what the data looks like, what stands out, what is concerning
- **Backstory**: A business intelligence communicator who bridges the gap between raw statistics and human-readable insight, specialising in making data findings accessible without losing accuracy
- **Tools**: None — this agent reasons purely from the outputs of previous agents
- **LLM**: `ollama/phi3:mini`

### Agent 4 — Anomaly Agent
- **Role**: Data Quality Inspector
- **Goal**: Identify every data quality red flag in the dataset — impossible values, placeholder strings, outliers, inconsistencies — and produce a clear actionable list of issues for a data engineer to fix
- **Backstory**: A detail-obsessed data quality engineer who has debugged production pipelines and knows that silent data errors cause the most damage — the kind of person who checks every assumption
- **Tools**: Outlier Detection Tool
- **LLM**: `ollama/llama3.2:3b`

### Agent 5 — Report Agent
- **Role**: Technical Report Writer
- **Goal**: Assemble the outputs of all previous agents into a single, coherent, well-structured EDA report with clear sections, and write it to disk
- **Backstory**: A technical writer with a data science background who knows how to organise complex findings into a report that is both rigorous and readable — the kind of document you would share with a client or senior stakeholder
- **Tools**: Report Writer Tool
- **LLM**: `ollama/phi3:mini`

---

## Part 5 — Define the 5 Tasks

### Task 1 — Profile the Dataset (assigned to Ingestion Agent)
- **Description**: Call the Dataset Profile Tool on retail_sales_dirty.csv. Return a structured summary including: all column names and their current stored dtypes, unique value counts, missing percentage per column, and 5 representative sample rows. Do not interpret or clean — only profile and report facts.
- **Expected output**: A structured text block with one section per column covering dtype, unique count, missing %, and a sample values list, followed by 5 sample rows.

### Task 2 — Compute Statistics (assigned to Statistical Agent)
- **Description**: Using the dataset profile from Task 1 as context, call the Statistical Summary Tool. Compute descriptive statistics for all numeric columns, value frequency counts for all categorical columns, a correlation matrix, and the target column distribution. Return all results as structured text organised by column.
- **Expected output**: A structured statistics report with numeric summaries, categorical summaries, a correlation table, and target distribution.
- **Context**: Task 1's output

### Task 3 — Interpret the Findings (assigned to Interpretation Agent)
- **Description**: Using the statistical output from Task 2, write a plain-English EDA narrative covering: what kind of data this appears to be and what business process it represents; which features appear most related to the target; whether there is class imbalance worth addressing; which pairs of features appear redundant; and what the 3 most important takeaways are for a data scientist about to build a model on this data.
- **Expected output**: 4-6 paragraphs of plain-English interpretation, ending with a bulleted "Key Takeaways" section of exactly 3 points.
- **Context**: Task 1 and Task 2 outputs

### Task 4 — Identify Anomalies (assigned to Anomaly Agent)
- **Description**: Call the Outlier Detection Tool. For every column, report the number and nature of outliers found by both Z-score and IQR methods, identify any placeholder/error strings, and flag any business-logic violations (e.g. negative quantities, future transaction dates). For each flag, write a one-line plain-English description of the issue and a recommended action (remove, cap, replace, or investigate further).
- **Expected output**: A bulleted anomaly report, one section per column, each with: outlier count, type of anomaly, plain-English description, recommended action.
- **Context**: Task 1's output

### Task 5 — Write the Final Report (assigned to Report Agent)
- **Description**: Assemble the outputs of Tasks 1 through 4 into a single, well-structured EDA report with the following sections in order: (1) Dataset Overview, (2) Statistical Summary, (3) Key Findings and Interpretation, (4) Data Quality and Anomaly Report, (5) Recommendations for Data Preparation before Modeling. Write the complete report as markdown, then call the Report Writer Tool to save it to the outputs/ folder.
- **Expected output**: A complete markdown EDA report saved to disk, plus a confirmation message with the filename.
- **Context**: All previous task outputs

---

## Part 6 — Assemble and Run the Crew

1. In `main.py`, import your agents and tasks, then create a `Crew` object with:
   - `agents`: list of all 5 agents in order
   - `tasks`: list of all 5 tasks in order
   - `process`: set to `Process.sequential` — tasks run one after another, each receiving the previous task's output as context
   - `verbose`: set to `True` during development — this prints every agent's reasoning and action to the terminal, which is essential for debugging and is also great content for your blog post
2. Call `crew.kickoff()` to run the full pipeline.
3. Monitor the terminal output — you will see each agent's reasoning steps, tool calls, and outputs printed in real time.

---

## Part 7 — Run for Both Models

1. Run the full crew with `llama3.2:3b` as the LLM for all 5 agents. Save the terminal output and the generated report.
2. Switch the LLM in all agent definitions to `phi3:mini` and run again. Save separately.
3. Note: you can also mix models per agent (e.g. Interpretation Agent and Report Agent on phi3:mini, others on llama3.2:3b) and compare mixed-model crews vs. single-model crews as an additional comparison dimension.

---

## Part 8 — Comparison and Evaluation

Build an evaluation table comparing the two model runs on:
- **Completeness**: did every section of the EDA report get populated, or did any agent produce an empty/partial output?
- **Statistical accuracy**: spot-check 5 statistics from the report against your own pandas computation — how many are correct?
- **Interpretation quality**: is the plain-English narrative specific to this dataset, or generic boilerplate? (subjective 1–5 score)
- **Anomaly detection recall**: how many of the known anomalies in this dataset did the Anomaly Agent catch? (you already know the issues from your earlier copilot project)
- **Report structure**: does the final report follow the requested sections in order?
- **Total wall-clock time**: end-to-end pipeline time per model run
- **Fallback or error count**: how many times did an agent fail to call a tool correctly or produce unparseable output?

---

## Part 9 — Article Structure
1. **Hook**: "I replaced my EDA notebook with a 5-agent CrewAI system. Here's the report it produced — and where it hallucinated."
2. **What CrewAI is**: one paragraph on the framework and why role-based agents make sense for EDA (each step genuinely needs a different specialisation).
3. **The agent team**: introduce each agent with its role and tool — use a simple diagram of the handoff chain.
4. **The report**: show the actual generated EDA report (or key excerpts) from the best-performing model run.
5. **The accuracy check**: show your spot-check table — which stats were right, which were wrong.
6. **The anomaly miss**: show at least one anomaly your Anomaly Agent missed that you found manually — honesty here is what makes the post credible.
7. **Model comparison**: llama3.2:3b vs phi3:mini — which produced a better report and why.
8. **Verdict**: is a 5-agent EDA system useful, or is `pandas-profiling` still faster and more reliable for this specific task?
