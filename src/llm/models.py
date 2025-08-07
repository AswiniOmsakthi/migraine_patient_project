import os

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


# # Set up Azure OpenAI
def azure_openai_chat_model(params: dict):
    return AzureChatOpenAI(
        azure_deployment=params['engine'],
        api_version="2023-05-15",
        temperature=params['temperature'],
        max_tokens=params['max_tokens'],
        top_p=params['top_p'],
        frequency_penalty=params['frequency_penalty'],
        presence_penalty=params['presence_penalty'],
        timeout=120,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]    # type: ignore
    )


# # Set up embeddings
def azure_openai_embedding_model(model: str = "text-embedding-3-large"):
    return AzureOpenAIEmbeddings(model=model)