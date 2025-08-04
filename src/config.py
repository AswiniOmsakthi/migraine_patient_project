###############################
# Azure OpenAI API Setup      #
###############################

OPENAI_API_CONFIG = {
    "OPENAI_API_TYPE": "azure",
    "OPENAI_API_VERSION": "2024-12-01",
    "OPENAI_API_REGION": "eastus2",
    "OPENAI_API_ENVIRONMENT": "dev",
}

# Chat model parameters
CHAT_MODEL_PARAMETERS: dict = {
    "max_tokens": 8000,
    "temperature": 0.0,
    "top_p": 0.05,
    "frequency_penalty": 0.0,
    "presence_penalty": 0,
    "stop": None,
    "engine": "gpt-4.1-mini"
}

###########################################
# Local Paths for Vector Store & Metadata #
###########################################

# Folder where Chroma vector database is stored
vector_store_path = "migraine_patient/chroma_data"

# Excel file mapping Aimovig PDF files with friendly names/descriptions
vector_store_metadata_file_path = "migraine_patient/src/data_processing/smpc_aimovig_filename_description.xlsx"

# Migraine interview Q&A CSV file
migraine_qa_file_path = "migraine_patient/product_info_pdfs/migraine interview question answer.csv"

# Directory containing 14 Aimovig product information PDFs
aimovig_pdf_dir = "migraine_patient/product_info_pdfs"

# Metadata file path for all PDFs (used for tool ingestion or labeling)
pdf_metadata_file_path = "migraine_patient/src/data_processing/smpc_aimovig_filename_description.xlsx"

##########################
# Streamlit App Content  #
##########################

page_title = "Synthetic AI Persona (Migraine Patient)"

conversation_starter = "Hi! I'm Lars. I’ve lived with migraine and used Aimovig — happy to talk if you're curious."

# Avatar image of the AI persona (larrs)
avatar_image_path = "src/images/lars.png"

# Optional branding (can be replaced)
bi_logo_url = "https://www.boehringer-ingelheim.com/sites/default/files/2024-04/Boehringer_Ingelheim_Accent-Green_0.png"

about_lars = """
# About Lars
Lars isn’t just an AI — he’s a voice shaped by lived migraine experience.
For over 30 years, Lars endured chronic daily migraines — the pain, the missed moments, and the emotional toll.
Now in his mid-50s, life took a new turn after starting Aimovig (erenumab).
It didn’t solve everything, but it gave back something vital: hope, and more good days.

Lars brings that personal journey to every conversation — the neurologist visits, the family impact, and the quiet resilience it takes to keep going.
He speaks with honesty, not authority — offering reflections, not medical advice.
His goal? To help others feel seen, understood, and supported in their own migraine story.
"""


##########################
# Tool Restrictions       #
##########################

# Tools with limited usage or requiring special handling
restricted_tools = ['pubmed_docs']
