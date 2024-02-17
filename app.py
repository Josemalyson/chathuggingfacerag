import random
import time
import os

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 500},
    huggingfacehub_api_token=huggingfacehub_api_token
)

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt_input := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        template = """      
            System: Você é um assistente virtual e ajude o usuário respondendo todas as perguntas dele.
            Responda no idioma portugues brasileiro.
            
            User: {question}      
            
            Assistant:
            """

        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        rag_chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        response = rag_chain.invoke(prompt_input)
        st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
