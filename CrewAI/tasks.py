from crewai import Task
from tools import youtube_tool
from agents import blog_researcher, blog_writer


### Research Task

research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get the detailed information about the video from the channel."
    ),
    expected_output="A comprehensive 3 paragraphs long research report based on the {topic} of video content.",
    tools=[youtube_tool],
    agents=[blog_researcher],
)

### Writing Task
writing_task = Task(
    description=(
        "get the info from the youtube channel on the topic {topic}."
    ),
    expected_output="Summarize the info from the youtube channel video on the topic {topic}",
    tools=[youtube_tool],
    agents=[blog_writer],
    async_execution=False,
    output_file="output/blog_post.txt",
)

