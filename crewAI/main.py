from agents import Agents
from tasks import BlogTasks

from crewai import Crew
import streamlit as st


class NewsCrew:
    def __init__(self, location, topic, language="english"):
        self.location = location
        self.language = language
        self.topic = topic

    def run(self):
        agents = Agents()
        tasks = BlogTasks()

        topic_chooser = agents.topic_chooser(self.location, self.topic, self.language)
        news_researcher = agents.researcher(self.location, self.topic, self.language)
        writer = agents.writer(self.location, self.topic, self.language)
        editor_in_chief = agents.editor_in_chief(self.location, self.topic, self.language)

        identify_stories = tasks.identify_stories(topic_chooser, self.location, self.topic, self.language)
        find_information = tasks.reaserch_topic(news_researcher, self.location, self.topic, self.language)
        write_blog_post = tasks.write_blog_post(writer, self.location, self.topic, self.language)
        edit_and_review_content = tasks.edit_and_review_content(editor_in_chief, self.location, self.topic,
                                                                self.language)

        crew = Crew(
            agents=[
                topic_chooser,
                news_researcher,
                writer,
                editor_in_chief,
            ],
            tasks=[
                identify_stories,
                find_information,
                write_blog_post,
                edit_and_review_content,
            ],
            verbose=True
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    st.subheader("Let AI Create a news article!",
                 divider="rainbow", anchor=False)

    with st.sidebar:
        st.header("What kind of news do you want")
        with st.form("my_form"):
            topic = st.text_area("What topic should the news cover?",
                                 placeholder="science")
            location = st.text_input(
                "For which location?", placeholder="Belgium")
            language = st.text_input("In what language should the news be?",
                                     placeholder="English")
            submitted = st.form_submit_button("Submit")

        st.divider()

if submitted:
    with st.status("The team is working on your news summary...", state="running", expanded=True) as status:
        with st.container(height=500, border=False):
            crew = NewsCrew(location, topic, language)
            result = crew.run()
        status.update(label="News Ready!",
                      state="complete", expanded=False)

    st.subheader("Here is your news summary:", anchor=False, divider="rainbow")
    st.markdown(result)
