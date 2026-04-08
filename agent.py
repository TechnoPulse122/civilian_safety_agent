import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import google.auth

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()
model_name = os.getenv("MODEL", "gemini-1.5-flash")

# --- CUSTOM TOOLS ---

def add_safety_request_to_state(tool_context: ToolContext, prompt: str) -> dict[str, str]:
    """Saves the user's safety concern or location to the shared state."""
    tool_context.state["SAFETY_PROMPT"] = prompt
    logging.info(f"[Safety State Update] User needs help with: {prompt}")
    return {"status": "success"}

# Configuring the Knowledge Tool (Wikipedia)
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# --- 1. SAFETY RESEARCHER AGENT ---
safety_researcher = Agent(
    name="safety_researcher",
    model=model_name,
    description="Researches humanitarian protocols and safety guidelines.",
    instruction="""
    You are a Safety Research Expert. Your goal is to gather information to keep a civilian safe.
    Analyze the user's SAFETY_PROMPT.
    - Use Wikipedia to find general safety protocols (e.g., 'how to find clean water', 'first aid for blast injuries').
    - Focus on data from reliable humanitarian organizations.
    - Synthesize the findings into a data block for the next agent.

    SAFETY_PROMPT:
    { SAFETY_PROMPT }
    """,
    tools=[wikipedia_tool],
    output_key="safety_research_data"
)

# --- 2. TASK & LOGISTICS AGENT ---
# This agent demonstrates "Task Management" by creating a checklist
task_coordinator = Agent(
    name="task_coordinator",
    model=model_name,
    description="Manages safety tasks, emergency schedules, and checklists.",
    instruction="""
    You are a Logistics Coordinator. Based on the SAFETY_RESEARCH_DATA, create a specific 
    step-by-step Task List for the user.
    - If they are evacuating, list items to pack.
    - If they are sheltering, list structural safety checks.
    - Organize these tasks by priority (Immediate vs. Secondary).

    RESEARCH_DATA:
    { safety_research_data }
    """,
    output_key="task_list"
)

# --- 3. EMERGENCY FORMATTER AGENT ---
emergency_formatter = Agent(
    name="emergency_formatter",
    model=model_name,
    description="Formats information into high-clarity emergency instructions.",
    instruction="""
    You are the Emergency Response Voice. Your task is to take the TASK_LIST and 
    SAFETY_RESEARCH_DATA and present it to the civilian.
    - Use clear, calm, and direct language.
    - Use bold text for critical warnings.
    - Present the Task List first, followed by the research facts.
    - End with a message of calm support.

    TASK_LIST:
    { task_list }
    
    RESEARCH_DATA:
    { safety_research_data }
    """
)

# --- WORKFLOW ORCHESTRATION ---

safety_workflow = SequentialAgent(
    name="safety_workflow",
    description="The workflow for researching and coordinating civilian safety tasks.",
    sub_agents=[
        safety_researcher,  # Step 1: Research safety protocols
        task_coordinator,   # Step 2: Create a task/checklist
        emergency_formatter # Step 3: Format the final response
    ]
)

root_agent = Agent(
    name="safety_greeter",
    model=model_name,
    description="Entry point for the Civilian Safety Assistant.",
    instruction="""
    - Greet the user as a Safety Assistant designed to help during crises.
    - Ask them what their current situation is or what they need help with.
    - When they respond, use 'add_safety_request_to_state' to save their input.
    - Immediately transfer control to the 'safety_workflow'.
    """,
    tools=[add_safety_request_to_state],
    sub_agents=[safety_workflow]
)