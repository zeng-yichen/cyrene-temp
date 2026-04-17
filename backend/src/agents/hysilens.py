import os
from google import genai
from google.genai import types
from backend.src.db import vortex as P

class Hysilens:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or str(P.MEMORY_ROOT)
        
        # Initialize the new Gemini API Client
        # The new SDK automatically picks up the GEMINI_API_KEY environment variable.
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.gemini_api_key:
            self.gemini_client = genai.Client()
        else:
            self.gemini_client = None
            print("[HYSILENS WARNING]: GEMINI_API_KEY environment variable is not set.")

    def find_source(self, snippet, company, model_choice):
        """
        Scans the client's raw files and uses the specified LLM to determine
        which document the snippet conceptually originated from.
        """
        company_dir = os.path.join(self.base_dir, company)
        if not os.path.exists(company_dir):
            return "Error: Company directory not found."

        # 1. Gather all raw text documents (excluding the output folder)
        docs = {}
        for root, dirs, files in os.walk(company_dir):
            if 'output' in root.split(os.sep): 
                continue # Don't search the generated outputs
            
            for file in files:
                if file.endswith(('.txt', '.md', '.json', '.csv')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            docs[os.path.basename(filepath)] = f.read()
                    except Exception as e:
                        print(f"Hysilens skip: Could not read {file}")

        if not docs:
            return "No source documents found to search against in the client directory."

        # 2. Construct the evaluation prompt
        prompt = (
            "You are an analytical AI assistant. Read the following highlighted snippet "
            "and the provided source documents. Determine which source document(s) the "
            "information in the snippet was most likely extracted or synthesized from.\n\n"
            f"HIGHLIGHTED SNIPPET:\n\"{snippet}\"\n\n"
            "SOURCE DOCUMENTS:\n"
        )
        
        for name, content in docs.items():
            prompt += f"--- BEGIN {name} ---\n{content}\n--- END {name} ---\n\n"

        prompt += "Analyze the conceptual and factual overlap. Provide your answer detailing the most likely source document(s) and a 1-2 sentence explanation of why."

        # 3. Route to the correct model
        if "Gemini" in model_choice or "All" in model_choice:
            return self._call_gemini(prompt)
        elif "Claude" in model_choice:
            return self._call_claude(prompt)
        elif "GPT" in model_choice:
            return self._call_gpt(prompt)
        else:
            return self._call_gemini(prompt)

    # --- API Wrappers ---
    def _call_gemini(self, prompt):
        if not self.gemini_client:
            return "Configuration Error: GEMINI_API_KEY is not set. Please export it in your environment."
            
        try:
            print("Hysilens routing to Gemini 3.1 Pro via new SDK...")
            
            # Low temperature for highly factual, analytical tracing
            config = types.GenerateContentConfig(
                temperature=0.2, 
            )
            
            # Call the model using the updated client architecture
            response = self.gemini_client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt,
                config=config
            )
            return response.text
            
        except Exception as e:
            return f"Gemini API Error: {str(e)}"

    def _call_claude(self, prompt):
        print("Hysilens routing to Claude Opus 4...")
        return "Claude Opus 4 Analysis:\n(API call not yet implemented.)"

    def _call_gpt(self, prompt):
        print("Hysilens routing to GPT-5...")
        return "GPT-5 Analysis:\n(API call not yet implemented.)"