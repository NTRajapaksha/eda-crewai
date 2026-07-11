import os
import datetime
from crewai import Crew, Process
from agents import (
    ingestion_agent, 
    statistical_agent, 
    interpretation_agent, 
    anomaly_agent, 
    report_agent
)
from tasks import (
    task_profile,
    task_statistics,
    task_interpretation,
    task_anomaly,
    task_report
)

from dotenv import load_dotenv
load_dotenv()

def run_pipeline():
    print("Initializing Multi-Agent EDA Pipeline...")
    
    # Create the Crew
    eda_crew = Crew(
        agents=[
            ingestion_agent,
            statistical_agent,
            interpretation_agent,
            anomaly_agent,
            report_agent
        ],
        tasks=[
            task_profile,
            task_statistics,
            task_interpretation,
            task_anomaly,
            task_report
        ],
        process=Process.sequential,
        verbose=True,
        memory=False,
        cache=False, # Verbose output for terminal monitoring
        # --- FREE TIER LIMITATION HANDLING ---
        # If you are using a FREE tier API (like Groq's free tier), uncomment the line below.
        # This forces the pipeline to wait 60 seconds between tasks so your tokens-per-minute bucket resets.
        # max_rpm=1 
        # -------------------------------------
    )
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    print("Starting pipeline execution with Groq (Llama 3 70B)...")
    try:
        # Kickoff the crew
        result = eda_crew.kickoff()
        
        # Save the result natively to avoid LLM JSON parsing errors on massive markdown strings
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/eda_report_{timestamp}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(result))
            
        print("\n\nPipeline execution completed successfully!")
        print(f"Final Report saved to: {filename}")
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")

if __name__ == "__main__":
    run_pipeline()
