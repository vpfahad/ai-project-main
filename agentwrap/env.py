from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()


def set_service_api_values():
    """
    Loads and returns service API values from environment variables.
    Returns:
        dict: Dictionary containing SUPABASE_URL, REDIS_URI, and DATABASE_URI.
    """
    global SUPABASE_URL, REDIS_URI, DATABASE_URI
    SUPABASE_URL = os.getenv("SUPABASE_URL", None)
    REDIS_URI = os.getenv("REDIS_URI", None)
    DATABASE_URI = os.getenv("DATABASE_URI", None)
    return {
        "SUPABASE_URL": SUPABASE_URL,
        "REDIS_URI": REDIS_URI,
        "DATABASE_URI": DATABASE_URI,
    }


def set_environment_variables(
    provider: str = "google_genai",
    api_key: str = None,
    langchain_api_key: str = None,
    langchain_project: str = None,
    langchain_tracing_v2: str = "true",
    langchain_endpoint: str = "https://api.smith.langchain.com",
    azure_api_key: str = None,
    azure_endpoint: str = None,
) -> str:
    """
    Loads and sets required environment variables for the application.

    Args:
        provider (str): The API provider, e.g., 'google', 'openai', 'azureopenai', etc.
        api_key (str, optional): API key for the selected provider. Defaults to value from environment.
        langchain_api_key (str, optional): LangChain API key. Defaults to value from environment.
        langchain_project (str, optional): LangChain project name. Defaults to value from environment or 'genai-project'.
        langchain_tracing_v2 (str, optional): LangChain tracing flag.
        langchain_endpoint (str, optional): LangChain endpoint URL.
        azure_api_key (str, optional): Azure OpenAI API key. If not provided, will prompt if provider is 'azureopenai'.
        azure_endpoint (str, optional): Azure OpenAI endpoint URL. If not provided, will use default or prompt if provider is 'azureopenai'.

    Returns:
        str: Success message if environment variables are set successfully.
    """
    provider_env_map = {
        "google_genai": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "azureopenai": "AZURE_OPENAI_API_KEY",
        # Add more providers and their env variable names as needed
    }
    env_var = provider_env_map.get(provider.lower())
    if env_var and provider.lower() != "azureopenai":
        os.environ[env_var] = api_key or os.getenv(env_var, "")
    elif provider.lower() == "azureopenai":
        # Set Azure OpenAI API key
        azure_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["AZURE_OPENAI_API_KEY"] = azure_key

        # Set Azure OpenAI endpoint
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint

    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key or os.getenv(
        "LANGCHAIN_API_KEY", ""
    )
    os.environ["LANGCHAIN_PROJECT"] = langchain_project or os.getenv(
        "LANGCHAIN_PROJECT", "genai-project"
    )
    os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
    os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint

    return "✅ Environment variables loaded and set successfully."





