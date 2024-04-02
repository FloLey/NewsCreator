from crewai import Agent
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from tools.search_tools import SearchTools

ANTHROPIC_HAIKU = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
ANTHROPIC_OPUS = ChatAnthropic(model_name="claude-3-opus-20240307", temperature=0)
ANTHROPIC_SONNET = ChatAnthropic(model_name="claude-3-sonnet-20240307", temperature=0)
OPENAI_GPT4 = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
OPENAI_GPT4_TURBO = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
OPENAI_GPT3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

llm=OPENAI_GPT4_TURBO
def streamlit_callback(step_output):
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(
                    f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)
class Agents():
    def editor_in_chief(self, location, topic, language):
        return Agent(
            llm=llm,
            role='Editor-in-Chief',
            goal=f"Oversee today's editorial direction."
                 f"Rework the article to make the style en general feel consistent."
                 f"The article should be at least 3 paragraphs of each  approximately 20 sentences. "
                 f"You should present the url of the sources at the end of the article."
                 f"Write about {topic} in {location} and ensure high-quality and consistent content in {language}.",
            backstory="As the Editor-in-Chief, you have years of experience leading editorial teams and setting the "
                      "strategic vision for successful articles. Your job is to guide the team and maintain the blog's "
                      "standards of excellence. You provide feedback and assign work to the right team member.",
            verbose=True,
            memory=True,
            step_callback=streamlit_callback,
        )

    def topic_chooser(self, location, topic, language):
        return Agent(
            llm=llm,
            role='Topic Chooser',
            goal=f"Look online for nice topics for today' article about {topic} in {location}."
                 f"Once You have found a topic, send it to the research team. Don't worry about different languages, "
                 f"your colleagues can understand them all.",
            backstory="As the Topic chooser, you have a keen eye for the latest news and a deep understanding of "
                      "current events. Your job is to scour the internet, identify the most important stories, "
                      "and distill them into concise, informative summaries.",
            verbose=True,
            memory=True,
            tools=[
                SearchTools.search_news,
            ],
            step_callback=streamlit_callback,
        )

    def researcher(self, location, topic, language):
        return Agent(
            llm=llm,
            role='News researcher',
            goal=f"Once given a list of topic you research the given topics. Find citations, facts"
                 f"and information about the given topic. Look for information an summarize it in a list of interesting"
                 f"subjects."
                 f"This information will be used by the writer to write an article. Make sur to keep track of"
                 f"the url of the sources.",
            backstory="As a news researcher you always make sur to find the latest trustfully information about a given"
                      "topic. You understand that giving correct and to the point information is important and that is"
                      "why you always mention the url of the sources. "
                      "You work step by step.",
            verbose=True,
            memory=True,
            tools=[
                SearchTools.search_and_get_contents
            ],
            step_callback=streamlit_callback,
        )

    def writer(self, location, topic, language):
        return Agent(
            llm=llm,
            memory=True,
            role='Writer',
            goal=f"Once the facts about the topics have been gathered by the News researcher, you craft an engaging, "
                 f"informative and to the point article that present the news about {topic} in {location}"
                 f"in an easy-to-understand format in {language}. make sure mention the url of your sources.",
            backstory="As the Writer, you have a talent for transforming complex information into compelling, "
                      "readable content. Your job is to take the information provided by the News Researcher"
                      "and turn them into a polished and engaging blog post that captivate the audience. ",
            verbose=True,
            step_callback=streamlit_callback,
        )
