import streamlit as st
from final_pipeline import final_pipeline_script

def main():
    st.title("Evidence-Augmented LLMs for Misinformation Detection")

    # Custom CSS to style the app
    st.markdown("""
        <style>
            /* Change the background color of the entire page */
            
            /* Style the app title */
            .stApp .css-1f3it89 {
                font-weight: bold;
                font-size: 2em;
                text-align: center;
                margin-bottom: 1em;
            }
            /* Style the input boxes */
            .stTextInput>div>div>input {
                margin-bottom: 1em;
            }
            /* Style the error messages */
            .error-message {
                color: red;
                font-weight: bold;
            }
            /* Style the process button */
            .stButton>button {
                width: 100%;
                margin-top: 1em;
                height: 3em;
                font-size: 1em;
            }
        </style>
    """, unsafe_allow_html=True)

    # Input boxes for URL and Text
    st.markdown("This model has access to knowledge regarding news and political claims, so it's best at classifying topics related to politics and major events. Please note that this model can make mistakes, so it should only serve as a reference.")
    url = st.text_input("Enter URL (currently supporting CNN, FOX, NBC, CBS):", placeholder="https://example.com")
    text = st.text_area("Enter Text:", placeholder="Type or paste text here...", height=150)

    # Button to run the script
    if st.button("Fact Check!"):
        if url and text:
            st.markdown("<div class='error-message'>Please provide either a URL or text, not both.</div>", unsafe_allow_html=True)
        elif url or text:
            with st.spinner("Face Checking..."):
                if url:
                    final_pipeline_script(url=url, text=None)
                else:
                    final_pipeline_script(url=None, text=text)
        else:
            st.markdown("<div class='error-message'>Please provide either a URL or text.</div>", unsafe_allow_html=True)

    # Add a link at the bottom
    st.markdown("<div class='footer-link'><a href='https://seanjiang-0416.github.io/DSC-180B-website/' target='_blank'>Visit our Project website for more information</a></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
