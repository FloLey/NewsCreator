import os
from operator import itemgetter

from duckduckgo_search import DDGS

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from exa_py import Exa
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.legacy.postprocessor import FlagEmbeddingReranker

exa = Exa(api_key=os.environ["EXA_API_KEY"])
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

ANTHROPIC_HAIKU = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0, max_tokens=2048)
ANTHROPIC_OPUS = ChatAnthropic(model_name="claude-3-opus-20240307", temperature=0, max_tokens=2048)
ANTHROPIC_SONNET = ChatAnthropic(model_name="claude-3-sonnet-20240307", temperature=0, max_tokens=2048)
OPENAI_GPT4 = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, max_tokens=2048)
OPENAI_GPT4_TURBO = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, max_tokens=2048)
OPENAI_GPT3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, max_tokens=2048)


class DDGSRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str, *, callbacks=None, **kwargs: any) -> list[Document]:
        try:
            results = DDGS().news(query, max_results=10, region="be-fr", timelimit="w")
            documents = [Document(
                page_content=x['body'],
                metadata={
                    "title": x['title'],
                    "url": x['url'],
                    "date": x['date'],
                }
            ) for x in results]
        except Exception as e:
            print(f"Failed to get context: {e}")
            return []
        return documents


ddgs_retriever = DDGSRetriever()


class ExaRetriever(BaseRetriever):
    def get_relevant_documents(self, queries: str, *, callbacks=None, **kwargs: any) -> list[Document]:
        documents = []
        try:
            for q in queries:
                search_result = exa.search(f"{q['content']}", use_autoprompt=True, num_results=2)
                articles = search_result.results
                for article in articles:
                    article_content = exa.get_contents([article.id]).results[0].text.strip()
                    # Splitting the content on double new lines and filtering sections with a substantial amount of text
                    filtered_content_sections = [section for section in article_content.split("\n\n") if
                                                 len(section.split()) >= 50]
                    filtered_content = "\n\n".join(filtered_content_sections)
                    documents.append(Document(
                        page_content=filtered_content,
                        metadata={
                            "url": article.url,
                            "title": article.title if hasattr(article, 'title') else "No title available",
                        }
                    ))
        except Exception as e:
            print(f"Failed to get documents: {e}")
            return []
        return documents

exa_retriever = ExaRetriever()

# def get_relevant_news(input: dict):
#     topic = input['topic']
#     format_instructions = output_parser_list.get_format_instructions()
#     template = PromptTemplate(
#         template="""
#         articles:
#         {context}
#
#         Based on the articles above, generate a list of the 5 main and most relevant topics with the time period
#         they are relevant for. Only include topics related to {topic}
#
#         Here is an example output
#
#         1. Semiconductor research collaboration between South Korea and the European Union (EU) March 2024
#         2. Development of AI-enhanced beer by researchers in Belgium March 2024
#         3. Young Luxembourg scientist winning a silver medal at a science competition in Milan March 2024
#         4. Establishment of a joint business relationship between Sovos and PwC in Belgium to accelerate e-invoicing implementation March 2024
#         5. Belgium's ambition to maintain its leadership in the biotech industry and its alignment with the European Commission's global ambition March 2024
#
#         """
#         ,
#         input_variables=["topic"],
#         partial_variables={"format_instructions": format_instructions},
#     )
#
#     def format_docs(docs):
#         formatted_docs = []
#
#         for d in docs:
#             content = d.page_content
#             title = d.metadata.get("title", "No Title Available")
#             date = d.metadata.get("date", "No URL Available")
#
#             formatted_doc = f"Title: {title}\n\nDate: {date}\n\ncontent:\n{content}\n"
#             formatted_docs.append(formatted_doc)
#
#         return "\n_______________________________________________\n\n".join(formatted_docs)
#
#     chain = (
#             {"context": itemgetter("topic") | ddgs_retriever | format_docs, "topic": itemgetter("topic")}
#             | template
#             | ANTHROPIC_HAIKU
#             | output_parser_list
#     )
#     return json.dumps(chain.invoke({"topic": topic}))
#

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
    use_fp16=False
)


def choose_docs(docs: list, query: str):

    nodes = [NodeWithScore(
        node=TextNode(text=f"{doc.metadata['title']}: {doc.page_content}",
                      metadata={'url': doc.metadata['url'], 'title': doc.metadata['title']}), score=0) for doc in docs]

    # Create a QueryBundle with the original query
    query_bundle = QueryBundle(query_str=query)

    # Re-rank the nodes (documents)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)

    # Prepare the output, extracting title, date, content, and URL from the node and its metadata
    top_docs_info = [{
        "content": doc.text,
        "title": doc.metadata['title'],
        "url": doc.metadata['url']
    } for doc in ranked_nodes]

    return top_docs_info


def generate_articles(topic: str, location: str, language: str):
    topic = f"{topic} in {location}"
    template = PromptTemplate(
        template="""
articles:
{context_articles}

{context_additional}

Based on the articles above, generate a single article about {topic} in {language}

Only return the article starting with the title and an intro and finishing with a conclusion.
Be complete, and go into details. Focus and delivering facts. Below the article also give a list 
of all the sources you have used in a clickable format. 
An article without sources is worthless.
Make sure to add markdown tags to format the article. 
"""
        ,
        input_variables=["topic", "language"],
    )

    choose_docs_lambda = RunnableLambda(lambda args: choose_docs(args['docs'], args['query']))

    def format_docs(docs):
        formatted_docs = []
        for d in docs:
            # Check if the document is a dict or an object and extract content, title, and URL accordingly
            if isinstance(d, dict):
                content = d.get("content", "")
                url = d.get("url", "")
                title = d.get("title", "")
            else:
                content = getattr(d, "page_content", "")
                metadata = getattr(d, "metadata", {})
                url = metadata.get("url", "")
                title = metadata.get("title", "")

            # Formatting the document string to include the URL
            formatted_doc = f"Title: {title}\n\nURL: {url}\n\nContent:\n{content}\n"
            formatted_docs.append(formatted_doc)

        return "\n_______________________________________________\n\n".join(formatted_docs)

    format_docs_lambda = RunnableLambda(lambda x: format_docs(x))

    chain = (
            RunnablePassthrough.assign(
                base_docs=itemgetter("topic") | ddgs_retriever,
                topic=itemgetter("topic"),
                language=itemgetter("language"),
            )
            |
            RunnablePassthrough.assign(
                context_articles={
                     "docs": itemgetter('base_docs'),
                     "query": itemgetter("topic")
                 } | choose_docs_lambda,
                topic=itemgetter("topic"),
                language=itemgetter("language"),
            )
            |
            RunnablePassthrough.assign(
                context_articles=itemgetter("context_articles") | format_docs_lambda,
                context_additional=itemgetter("context_articles") | exa_retriever | format_docs_lambda,
                topic=itemgetter("topic"),
                language=itemgetter("language"),
            )
            | template
            | ANTHROPIC_HAIKU
    )


    return chain.invoke({"topic": f"{topic}", "language": language})
