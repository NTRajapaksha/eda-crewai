from crewai import Task
from agents import ingestion_agent, statistical_agent, interpretation_agent, anomaly_agent, report_agent

task_profile = Task(
    description='Call the Dataset Profile Tool EXACTLY ONCE, passing "retail_store_sales.csv" as the file_path argument. Immediately return its exact output as your final answer. DO NOT call the tool again. Do not interpret or clean — only profile and report facts.',
    expected_output='A structured text block with one section per column covering dtype, unique count, missing %, and a sample values list, followed by 5 sample rows.',
    agent=ingestion_agent
)

task_statistics = Task(
    description='Using the dataset profile from the previous task as context, call the Statistical Summary Tool EXACTLY ONCE, passing "retail_store_sales.csv" as the file_path argument. Immediately return its exact output as your final answer. DO NOT call the tool again. Return all results as structured text organised by column.',
    expected_output='A structured statistics report with numeric summaries, categorical summaries, a correlation table, and target distribution.',
    agent=statistical_agent,
    context=[task_profile]
)

task_interpretation = Task(
    description='Using the statistical output from the previous task, write a plain-English EDA narrative covering: what kind of data this appears to be and what business process it represents; which features appear most related to the target; whether there is class imbalance worth addressing; which pairs of features appear redundant; and what the 3 most important takeaways are for a data scientist about to build a model on this data. DO NOT output conversational filler like "Go ahead" or "Here is the answer". IMMEDIATELY begin with your narrative report.',
    expected_output='4-6 paragraphs of plain-English interpretation, ending with a bulleted "Key Takeaways" section of exactly 3 points.',
    agent=interpretation_agent,
    context=[task_profile, task_statistics]
)

task_anomaly = Task(
    description='Call the Outlier Detection Tool EXACTLY ONCE, passing "retail_store_sales.csv" as the file_path argument. Immediately return its exact output as your final answer. DO NOT call the tool again. For every column, report the number and nature of outliers found by both Z-score and IQR methods, identify any placeholder/error strings, and flag any business-logic violations (e.g. negative quantities, future transaction dates). For each flag, write a one-line plain-English description of the issue and a recommended action (remove, cap, replace, or investigate further).',
    expected_output='A bulleted anomaly report, one section per column, each with: outlier count, type of anomaly, plain-English description, recommended action.',
    agent=anomaly_agent,
    context=[task_profile]
)

task_report = Task(
    description='Assemble the outputs of the previous tasks into a single, well-structured EDA report with the following sections in order: (1) Dataset Overview, (2) Statistical Summary, (3) Key Findings and Interpretation, (4) Data Quality and Anomaly Report, (5) Recommendations for Data Preparation before Modeling. DO NOT use any tools. Simply return the complete markdown report as your exact and final answer. DO NOT output conversational filler like "Go ahead" or "Here is the report". IMMEDIATELY begin with your markdown report.',
    expected_output='A complete markdown EDA report saved to disk, plus a confirmation message with the filename.',
    agent=report_agent,
    context=[task_profile, task_statistics, task_interpretation, task_anomaly]
)
