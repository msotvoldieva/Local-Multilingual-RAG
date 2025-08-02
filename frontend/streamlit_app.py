import streamlit as st
import requests

st.title("📚 Galileo Local RAG")
query = st.text_input("Ask a question:")

if st.button("Submit"):
    response = requests.post("http://backend:8000/ask",
                        json={"question": query})
    
    data = response.json()
    st.markdown("### Answer")
    st.write(data.get('answer'))
