from typing import Any, Generator
from langchain_core.messages import HumanMessage


DEFAULT_MODELS = {
    "openai": ["gpt-3.5-turbo"],
    "google_genai": ["gemini-2.0-flash-001"],
    "mistral": ["mistral-large-latest"],
    "azureopenai": ["gpt-35-turbo"],
}


def load_chat_model(model_name: str = None, **kwargs) -> Any:
    """
    Load a chat model using LangChain wrappers.
    Supports OpenAI, Google, Mistral, and AzureOpenAI models.
    If no model_name is specified, uses the default model for each provider.

    Args:
        model_name (str, optional): Model identifier, e.g., "openai/gpt-4.1-mini", "google/gemini-pro", "mistral/mistral-7b", "azureopenai/gpt-35-turbo".

    Returns:
        Any: An instance of the loaded LangChain chat model.
    """
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mistralai import ChatMistralAI

    if model_name is None:
        raise ValueError(
            "No model_name specified. Please provide a model_name in the format 'provider/model-id', "
            "e.g., 'openai/gpt-3.5-turbo', or use 'openai', 'google', 'mistral', or 'azureopenai' to use the default."
        )

    if "/" not in model_name:
        provider = model_name.lower()
        if provider in DEFAULT_MODELS:
            model_id = DEFAULT_MODELS[provider][0]
            model_name = f"{provider}/{model_id}"
        else:
            raise ValueError(f"Unknown model provider '{provider}'.")

    if model_name.startswith("openai/"):
        model_id = model_name.split("/", 1)[1]
        return ChatOpenAI(model=model_id, **kwargs)
    elif model_name.startswith("google_genai/"):
        model_id = model_name.split("/", 1)[1]
        return ChatGoogleGenerativeAI(model=model_id, **kwargs)
    elif model_name.startswith("mistral/"):
        model_id = model_name.split("/", 1)[1]
        return ChatMistralAI(model=model_id, **kwargs)
    elif model_name.startswith("azureopenai/"):
        model_id = model_name.split("/", 1)[1]
        azure_params = {
            "azure_deployment": model_id,
            "api_version": kwargs.pop("api_version", "2023-06-01-preview"),
            "temperature": kwargs.pop("temperature", 0),
            "max_tokens": kwargs.pop("max_tokens", None),
            "timeout": kwargs.pop("timeout", None),
            "max_retries": kwargs.pop("max_retries", 2),
        }
        azure_params.update(kwargs)
        return AzureChatOpenAI(**azure_params)
    else:
        raise ValueError(f"Unknown model provider in '{model_name}'")


def load_configurable_chat_model(
    model: str = "google_genai:gemini-2.0-flash-001",
    temperature: float = 0,
    configurable_fields: tuple = (
        "model",
        "model_provider",
        "temperature",
        "max_tokens",
    ),
    config_prefix: str = "first",
):
    """
    Generalized function to create a configurable LLM using init_chat_model.

    Args:
        model (str): The model string, e.g., "google_genai:gemini-2.0-flash-001"
        temperature (float): The default temperature for the model.
        configurable_fields (tuple): Fields that can be configured at runtime.
        config_prefix (str): Prefix for config keys (useful for multiple models).

    Returns:
        An LLM instance with configurable fields.

    Example:
        >>> llm = load_configurable_chat_model(
        ...     model="google_genai:gemini-2.0-flash-001",
        ...     temperature=0.5,
        ...     configurable_fields=("model", "temperature", "max_tokens"),
        ...     config_prefix="second"
        ... )
        >>> response = llm.invoke(
        ...     "what's your name",
        ...     config={
        ...         "configurable": {
        ...             "second_model": "claude-3-5-sonnet-2024062",
        ...             "second_temperature": 0.5,
        ...             "second_max_tokens": 100,
        ...         }
        ...     },
        ... )
        >>> print(response.content)
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=model,
        temperature=temperature,
        configurable_fields=configurable_fields,
        config_prefix=config_prefix,
    )


def ask_llm(
    question: str,
    model_name: str = "google_genai/gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
) -> str:
    """
    Simple function to ask a question to an LLM and get a beautified, structured response.

    Args:
        question (str): The question to ask the LLM
        model_name (str): Model identifier (e.g., "openai/gpt-3.5-turbo", "google/gemini-pro")
        temperature (float): Controls randomness in the response (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens in the response
        **kwargs: Additional arguments to pass to the model

    Returns:
        str: A beautified message with the user question and the LLM's answer in a structured format

    Example:
        response = ask_llm("What is the capital of France?")
        print(response)
    """
    # Load the model
    model = load_chat_model(
        model_name, temperature=temperature, max_tokens=max_tokens, **kwargs
    )

    # Create the message
    message = HumanMessage(content=question)

    # Get response
    response = model.invoke([message])

    # Beautified structured output
    beautified = (
        "====================\n"
        "🤔 User Question:\n"
        f"{question}\n"
        "--------------------\n"
        "💡 LLM Answer:\n"
        f"{response.content}\n"
        "===================="
    )
    return beautified


def ask_llm_stream(
    question: str,
    model_name: str = "google_genai/gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: int = 100,
    stream_prefix: str = "💡 LLM Answer:\n",
    **kwargs,
) -> Generator[str, None, None]:
    """
    Streaming version of ask_llm that yields response chunks as they arrive.

    Args:
        question (str): The question to ask the LLM
        model_name (str): Model identifier (e.g., "openai/gpt-3.5-turbo", "google/gemini-pro")
        temperature (float): Controls randomness in the response (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens in the response
        stream_prefix (str): Prefix to add before streaming the response
        **kwargs: Additional arguments to pass to the model

    Yields:
        str: Chunks of the response as they arrive

    Example:
        for chunk in ask_llm_stream("What is machine learning?"):
            print(chunk, end='', flush=True)
    """
    # Load the model
    model = load_chat_model(
        model_name, temperature=temperature, max_tokens=max_tokens, **kwargs
    )

    # Create the message
    message = HumanMessage(content=question)

    # Print the question header
    yield (
        "====================\n"
        "🤔 User Question:\n"
        f"{question}\n"
        "--------------------\n"
        f"{stream_prefix}"
    )

    # Stream the response
    try:
        for chunk in model.stream([message]):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"\n❌ Error during streaming: {str(e)}"

    # Print the closing separator
    yield "\n===================="


def interactive_llm_chat(
    model_name: str = "google_genai/gemini-2.0-flash-001",
    temperature: float = 0.2,
    max_tokens: int = 100,
    welcome_message: str = None,
    **kwargs,
) -> None:
    """
    Interactive console chat with LLM using streaming responses.
    Users can ask multiple questions until they type 'exit'.

    Args:
        model_name (str): Model identifier (e.g., "openai/gpt-3.5-turbo", "google/gemini-pro")
        temperature (float): Controls randomness in the response (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens in the response
        welcome_message (str): Custom welcome message (optional)
        **kwargs: Additional arguments to pass to the model

    Example:
        interactive_llm_chat(model_name="openai/gpt-4")
    """
    if welcome_message is None:
        welcome_message = (
            "🤖 Interactive LLM Chat\n"
            "====================\n"
            "Ask me anything! Type 'exit' to quit.\n"
            "Type 'help' for available commands.\n"
            "====================\n"
        )

    print(welcome_message)
    from src.agentwrap.env import set_environment_variables

    status = set_environment_variables()
    print(status)

    # Load the model once for efficiency
    try:
        model = load_chat_model(
            model_name, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        print(f"✅ Model loaded: {model_name}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    question_count = 0

    while True:
        try:
            # Get user input
            print(f"\n💬 Question #{question_count + 1}: ", end="")
            user_input = input().strip()

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye! Thanks for chatting!")
                break

            # Check for help command
            if user_input.lower() in ["help", "h"]:
                print("\n📚 Available Commands:")
                print("  - Type your question to get an answer")
                print("  - 'exit', 'quit', or 'q' to exit")
                print("  - 'help' or 'h' to show this help")
                print("  - 'clear' or 'c' to clear the screen")
                print("  - 'model' or 'm' to show current model info")
                continue

            # Check for clear command
            if user_input.lower() in ["clear", "c"]:
                import os

                os.system("cls" if os.name == "nt" else "clear")
                print(welcome_message)
                continue

            # Check for model info command
            if user_input.lower() in ["model", "m"]:
                print(f"\n🔧 Current Model: {model_name}")
                print(f"🌡️  Temperature: {temperature}")
                print(f"📏 Max Tokens: {max_tokens}")
                continue

            # Skip empty input
            if not user_input:
                print("❌ Please enter a question.")
                continue

            # Process the question
            question_count += 1
            print(f"\n🤔 Processing question #{question_count}...")

            # Create the message
            message = HumanMessage(content=user_input)

            # Stream the response
            print("\n💡 Answer:")
            print("--------------------")

            try:
                response_chunks = []
                for chunk in model.stream([message]):
                    if hasattr(chunk, "content") and chunk.content:
                        chunk_content = chunk.content
                        response_chunks.append(chunk_content)
                        print(chunk_content, end="", flush=True)

                print("\n--------------------")
                print(
                    f"✅ Response completed ({len(''.join(response_chunks))} characters)"
                )

            except Exception as e:
                print(f"\n❌ Error during streaming: {str(e)}")
                print("Please try again.")

        except KeyboardInterrupt:
            print(
                "\n\n⚠️  Interrupted by user. Type 'exit' to quit or continue asking questions."
            )
        except EOFError:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again.")


def prompt_executor(
    user_prompt: str,
    llm,
    system_prompt: str = None,
    input_variables: dict = None,
    async_mode: bool = False,
    runnable_config: dict = None,
    return_chain: bool = False,
):
    """
    Flexible function to get a summary (or any LLM response) or just the chain.

    Args:
        user_prompt (str): The user prompt (can include {variables} for formatting). Required.
        llm: The language model instance (should support .invoke or .ainvoke).
        system_prompt (str, optional): The system prompt for the LLM. If not provided, uses a default.
        input_variables (dict, optional): Variables to format the user prompt. Defaults to None.
        async_mode (bool, optional): If True, uses .ainvoke for async LLMs. Defaults to False.
        runnable_config (dict or RunnableConfig, optional): Optional config for the LLM chain.
        return_chain (bool, optional): If True, return the chain object instead of running it.

    Returns:
        The LLM's response (str) or the chain object if return_chain=True.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableConfig

    # Use default system prompt if not provided
    if not system_prompt:
        system_prompt = "You are a helpful assistant."

    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )
    chain = chat_prompt | llm | StrOutputParser()
    if return_chain:
        return chain

    input_variables = input_variables or {}
    runnable_config = runnable_config or {}

    if async_mode:
        return chain.ainvoke(input_variables, config=runnable_config)
    else:
        return chain.invoke(input_variables, config=runnable_config)


if __name__ == "__main__":
    # Run the demo if this file is executed directly
    interactive_llm_chat()
