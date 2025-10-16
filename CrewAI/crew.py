from crewai import Crew, Process
from agents import blog_researcher, blog_writer
from tasks import research_task, writing_task


# Define the Crew with tasks and agents
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writing_task],
    process=Process.SEQUENTIAL, # Optional: sequential task is default or parallel
    name="YouTube Blog Creator",
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True,
)

### start the task execution process
result = crew.kickoff(inputs={"topic":"Flood Prediction"})
print(result)