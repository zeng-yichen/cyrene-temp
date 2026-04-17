import os
from google import genai
from google.genai import types
from backend.src.db import vortex as P


class Cerydra:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or str(P.MEMORY_ROOT)
        os.makedirs(self.base_dir, exist_ok=True)

        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client()
        else:
            self.client = None
            print("[CERYDRA WARNING]: GEMINI_API_KEY environment variable is not set.")

    def get_file_tree(self):
        """Returns a nested dictionary representing the directory structure."""
        tree = {}
        for root, dirs, files in os.walk(self.base_dir):
            rel_path = os.path.relpath(root, self.base_dir)
            if rel_path == ".":
                current_node = tree
            else:
                parts = rel_path.split(os.sep)
                current_node = tree
                for part in parts:
                    current_node = current_node.setdefault(part, {})

            for file in files:
                if not file.startswith('.'):
                    current_node[file] = os.path.join(root, file)
        return tree

    def read_document(self, filepath):
        """Safely reads and returns the text content of a document."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file:\n{e}"

    def query_documents(self, company, question, draft_text=None):
        """
        Gathers all client documents and queries the LLM against them.
        Perfect for asking "Why did this draft fail?" or "What is their brand voice?"
        """
        if not self.client:
            return "Configuration Error: GEMINI_API_KEY is not set in your environment."

        company_dir = os.path.join(self.base_dir, company)
        if not os.path.exists(company_dir):
            return f"Error: No data directory found for company '{company}'."

        # 1. Gather context from all raw client files
        context_docs = ""
        for root, dirs, files in os.walk(company_dir):
            if 'output' in root.split(os.sep): 
                continue # Skip generated outputs so it doesn't grade its own homework
            
            for file in files:
                if file.endswith(('.txt', '.md', '.json', '.csv')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            context_docs += f"--- BEGIN {file} ---\n{content}\n--- END {file} ---\n\n"
                    except Exception as e:
                        print(f"Cerydra skip: Could not read {file}")

        if not context_docs.strip():
            return "No source documents found to analyze."

        # 2. Construct the evaluation prompt
        prompt = (
            "You are Cerydra, an expert client strategist and document analyst. "
            "Review the provided client source documents. Then, answer the user's question "
            "using ONLY the context provided by these documents. If a draft is provided, "
            "evaluate it strictly against the client's stated preferences, tone guidelines, and past feedback.\n\n"
            f"SOURCE DOCUMENTS:\n{context_docs}\n"
        )
        
        if draft_text:
            prompt += f"DRAFT TO EVALUATE:\n\"{draft_text}\"\n\n"
            
        prompt += f"USER QUESTION:\n{question}\n"

        # 3. Call the model via the new SDK
        try:
            print(f"[CERYDRA] Querying documents for {company}...")
            # Temperature is kept relatively low (0.3) so the model focuses on factual 
            # evidence in the client documents rather than inventing advice.
            config = types.GenerateContentConfig(temperature=0.3)
            
            response = self.client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt,
                config=config
            )
            return response.text
            
        except Exception as e:
            return f"Query Error: {str(e)}"