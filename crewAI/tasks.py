from textwrap import dedent

from crewai import Task


class BlogTasks():
    def edit_and_review_content(self, agent, location, topic, language):
        return Task(
            description=dedent(f""" 
        Rework the article to make the style en general feel consistent.
        The article should be at least 3 paragraphs of each  approximately 20 sentences.
        There should also be a title, an intro and a conclusion.          
        Your final output should be the edited article posts in {language}.
        make sure the url of the sources are present at the end of the article.
            """),
            agent=agent,
            expected_output=f"A completed article ready for publication covering the news for {topic} in {location}"
                            f"in {language} with the url of the sources. The article should have a title, an intro and "
                            f"a conclusion and a least 3 paragraphs of each approximately 20 sentences"
        )

    def identify_stories(self, agent, location, topic, language):
        return Task(
            description=dedent(f"""
                Identify some interesting news stories for today's article about {topic} in {location}. 
                Provide a summary of the subject in {language}.
            """),
            agent=agent,
            expected_output=f"A summary of the interesting story that can be used in "
                            f"today's news. do not include the source or their url."
        )

    def reaserch_topic(self, agent, location, topic, language):
        return Task(
            description=dedent(f"""
                Given a list of subject gather information about the subjects. Gather relevant information to cover the topic
                to help the writer write an article. Gather information/news about the situations, people, places, ...
                that are relevant to the subject. You gather a list of facts, citations, ... with the url of the sources that
                you think can be usefully for the writer to write an article.
            """),
            agent=agent,
            expected_output=f"A summary containing a list of information (facts, citations, ...) "
                            f"and the urls of the sources about the subjects in {language}"
        )

    def write_blog_post(self, agent, location, topic, language):
        return Task(
            description=dedent(f"""
                Write an engaging article about the given topics. The article should be
                well-structured, with a clear introduction and
                conclusion. The body should describe each subject in at least 20 sentences.
                If you are laking information, ask for more information to the research team.
                make sure to include the urls of the sources at the end of the article.
                Your final output should be the article ready for review in {language}.
            """),
            agent=agent,
            expected_output="A completed article with at least 3 paragraphs of 20 sentences ready "
                            "for review with the urls of the sources."
        )
