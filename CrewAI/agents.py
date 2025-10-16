from crewai import Agent
from tools import youtube_tool
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

llm = LLM(model="gemini-2.5-flash")

### Create a senior blog content researcher

blog_researcher = Agent(
    name="Blog Researcher from Youtube Videos",
    goal="get the revelant video content for the topic {topic} from YT channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, Machine Learning and GEN AI provide suggestions for blog content"
    ),
    llm=llm,
    tools=[youtube_tool],
    allow_delegation=True,
)


#### Create a senior blog content writer

blog_writer = Agent(
    name = "Blog Writer",
    goal="Write a detailed blog post on the topic {topic} using the research content provided by the researcher",
    verbose=True,
    llm=llm,
    memory=True,
    backstory=(
        "With the flair for simplifying complex topics, you craft engaging and informative blog posts that captivate readers and enhance their understanding."
    ),
    tools=[youtube_tool],
    allow_delegation=False,
)

