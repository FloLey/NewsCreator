import streamlit as st

from news_generator import generate_articles

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
        with st.status("The news article is being created...", state="running", expanded=True) as status:
            with st.container(height=10, border=False):
                result = generate_articles(topic, location, language)
            status.update(label="News Ready!", state="complete", expanded=False)

        st.subheader("Here is your news summary:", anchor=False, divider="rainbow")
        st.markdown(result.content)
