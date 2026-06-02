import os
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langfuse import Langfuse

# Correct Langfuse Import
from langfuse.langchain import CallbackHandler

# =========================
# Load Environment Variables
# =========================
load_dotenv()

# =========================
# API KEYS
# =========================
google_api_key = os.getenv("GOOGLE_API_KEY")

# =========================
# Langfuse Setup
# =========================
langfuse=Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

langfuse_handler = CallbackHandler()
# =========================
# Gemini Model
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key,
    temperature=0.3
)

print("Gemini + Langfuse Connected Successfully!")
# =========================
# Prompt Template
# =========================
prompt = ChatPromptTemplate.from_template("""
You are an AI Code Review Assistant.

Analyze ONLY the given code snippet.
Do NOT execute code.

Provide output strictly in JSON format.

Expected JSON Schema:
{{
    "identified_issues": ["string"],
    "improvement_suggestions": ["string"],
    "code_quality_level": "string",
    "review_summary": "string"
}}

Code Snippet:
{code}
""")

# =========================
# JSON Parser
# =========================
parser = JsonOutputParser()

# =========================
# Create Chain
# =========================
chain = prompt | llm | parser

# =========================
# User Input
# =========================
code_input = st.text_area(
    "Paste Your Code Here 👇",
    height=300
)

# =========================
# Generate Review
# =========================
if st.button("🚀 Generate Review"):

    if code_input.strip() == "":
        st.warning("Please enter some code.")

    else:

        with st.spinner("Analyzing code..."):

            try:

                response = chain.invoke(
                    {
                        "code": code_input
                    },
                    config={
                        "callbacks": [langfuse_handler]
                    }
                )
                
                langfuse.flush()
                st.success("Review Generated Successfully!")

                # =========================
                # Issues
                # =========================
                st.subheader("📌 Identified Issues")

                for issue in response["identified_issues"]:
                    st.error(issue)

                # =========================
                # Suggestions
                # =========================
                st.subheader("✅ Improvement Suggestions")

                for suggestion in response["improvement_suggestions"]:
                    st.info(suggestion)

                # =========================
                # Quality
                # =========================
                st.subheader("📊 Code Quality Level")
                st.write(response["code_quality_level"])

                # =========================
                # Summary
                # =========================
                st.subheader("📝 Review Summary")
                st.write(response["review_summary"])

                # =========================
                # Raw JSON
                # =========================
                st.subheader("📦 Raw JSON Output")
                st.json(response)

            except Exception as e:
                st.error(f"Error: {str(e)}")