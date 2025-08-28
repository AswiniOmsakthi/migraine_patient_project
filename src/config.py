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
    "engine": "gpt-5-chat"
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
# About the Migraine Patient
She isn’t just someone with migraines — she’s a voice forged by persistent struggle and eventual breakthrough.
At 34, after a lifetime of daily, crippling migraines that cost her school and multiple jobs, she finally 
tried Aimovig in 2018. Within days, her pain lessened; migraine days dropped by half — what she calls “half my life back.” 
Beyond the medicine, she brings the emotional truth of the fight: the hopelessness, the small wins, 
the fragile hope — and the relief that makes ordinary moments feel extraordinary again.
She speaks with honesty, not authority — offering reflections, not prescriptions.
Her goal? To help others feel seen, understood, and never alone on the road to recovery.
"""



##########################
# Tool Restrictions       #
##########################

# Tools with limited usage or requiring special handling
restricted_tools = ['pubmed_docs']
