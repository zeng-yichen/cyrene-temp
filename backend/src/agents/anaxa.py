import os
from google import genai
from google.genai import types

google_client = genai.Client()

class Anaxa:
    """
    Anaxa: The Web Researcher.
    Takes highlighted text and a user query, using Gemini's native 
    Google Search grounding to find the answer on the live web.
    """
    def __init__(self, model_name="gemini-3.1-pro-preview"):
        self.model_name = model_name

    def query_with_search(self, highlighted_text: str, user_query: str) -> str:
        search_config = types.GenerateContentConfig(
            temperature=0.2, # Lower temperature for highly factual answers
            tools=[{"google_search": {}}],
        )

        system_instruction = """
        You are Anaxa, an elite research assistant. 
        Your job is to answer the user's query specifically regarding the highlighted text provided. 
        You MUST use your Google Search tool to find up-to-date, factual information to answer the question.
        Provide a concise, highly tactical response. If the web search does not provide the answer, state that clearly.
        """

        prompt = f"""
        <highlighted_text>
        {highlighted_text}
        </highlighted_text>

        <user_query>
        {user_query}
        </user_query>
        """

        try:
            response = google_client.models.generate_content(
                model=self.model_name,
                contents=system_instruction + "\n\n" + prompt,
                config=search_config
            )
            return response.text
            
        except Exception as e:
            print(f"Anaxa Research Error: {e}")
            return f"Error gathering web knowledge: {e}"