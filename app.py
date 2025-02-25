# from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_loaders import SitemapLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate 
import streamlit as st
import os

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)



if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                
    Examples:
                                                
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

with st.sidebar:
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

    if st.button("Save API Key"):
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("API Key saved successfully!")
        else:
            st.error("Please enter a valid API Key.")

api_key = os.getenv("OPENAI_API_KEY", st.session_state.get("openai_api_key", ""))

if api_key :
    llm = ChatOpenAI(temperature=0.1)

    def get_answers(inputs):
        docs = inputs["docs"]
        question = inputs["question"]
        answers_chain = answers_prompt | llm
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, "context": doc.page_content}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
        }

    def choose_answer(inputs):
        answers = inputs["answers"]
        question = inputs["question"]
        choose_chain = choose_prompt | llm
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke(
            {
                "question": question,
                "answers": condensed,
            }
        )

    def parse_page(soup):
        header = soup.find("header")
        footer = soup.find("footer")
        if header:
            header.decompose()
        if footer:
            footer.decompose()
        return (
            str(soup.get_text())
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("CloseSearch Submit Blog", "")
        )

    @st.cache_resource(show_spinner="Loading website...")
    def load_website(url):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/workers-ai\/).*",
                r"^(.*\/vectorize\/).*",
                r"^(.*\/ai-gateway\/).*",
            ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 2
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        return vector_store.as_retriever()


    url = "https://developers.cloudflare.com/sitemap-0.xml"
    retriever = load_website(url)
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content)