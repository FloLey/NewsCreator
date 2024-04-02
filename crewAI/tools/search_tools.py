import os

from duckduckgo_search import DDGS
from langchain.tools import tool

from exa_py import Exa

exa = Exa(api_key=os.environ["EXA_API_KEY"])


class SearchTools:
    @tool()
    def search_news(query, top_result_to_return=5):
        """Useful to search news on the internet about a given topic and return relevant results"""
        results = DDGS().news(query, max_results=top_result_to_return, region="be-fr", timelimit="w")
        string = []
        for result in results:
            string.append('\n'.join([
                f"Title: {result['title']}", f"Link: {result['url']}",
                f"Snippet: {result['body']}", "\n-----------------"
            ]))
        return '\n'.join(string)

    @tool
    def search_and_get_contents(queries: list[str]):
        """
        Search for webpages based on a list of queries and return the content and URL of the top results.
        The queries should not contain references to URLs or website addresses.
        """
        results = []
        for query in queries:
            search_result = exa.search(f"{query}", use_autoprompt=True, num_results=3)
            articles = search_result.results
            for article in articles:
                article_content = exa.get_contents([article.id]).results[0].text.strip()
                # Splitting the content on double new lines and filtering sections with less than 5 words
                filtered_content_sections = [section for section in article_content.split("\n\n") if
                                             len(section.split()) >= 50]
                filtered_content = "\n\n".join(filtered_content_sections)
                results.append({"content": filtered_content, "url": article.url})

        # Formatting the output
        return '\n\n'.join([f"Content: {item['content']}, URL: {item['url']}" for item in results])
