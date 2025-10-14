
# PortOps RAG Chatbot — Streamlit Cloud Ready (v3)

This build includes a default `Port Operations Reference Manual.txt` that is auto-loaded at startup so the app is ready for demo immediately.

## How to deploy on Streamlit Cloud
1. Create a new GitHub repository and upload these files.
2. In Streamlit Cloud create a new app pointing to this repo and set `app.py` as the entrypoint.
3. (Optional) Add `OPENAI_API_KEY` in secrets if you want to use OpenAI instead of HF.

## Demo instructions
- The app loads the built-in manual on startup. Ask questions like:
  - `What are crane safety procedures?`
  - `List emergency contacts.`
  - `What weather restrictions are specified?`
\n