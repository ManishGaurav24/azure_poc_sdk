import asyncio
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

from utils.cosmos_connection import get_last_messages_from_cosmos
from utils.log_utils import debug_print

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("API_VERSION")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_key = os.getenv("SEARCH_KEY")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
Embedding_Endpoint = os.getenv("EMBEDDING_ENDPOINT")
index_name = os.getenv("INDEX_NAME")


async def warm_up_search_index():
    """
    Async warm up the search index with a simple query on app start
    """
    try:
        debug_print("Starting search index warmup...")
        # Make a simple query to warm up the index
        warmup_response = await call_llm_async_with_retry("What is this document about?",
                                                          "warmup-session",
                                                                    max_retries=1)
        debug_print(f"Warmup response: {warmup_response}")
        debug_print("Search index warmed successfully")
        return True
    except Exception as e:
        debug_print(f"Warmup failed: {str(e)}")
        return False


async def call_llm_async_with_retry(user_input: str, session_id: str, max_retries: int = 3, delay: int = 2):
    """
    Async call to LLM with retry logic, debug information, and Cosmos DB context
    """
    debug_print(f"User Query: {user_input}")
    print("API VERSION: ",api_version)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
    )

    # Enhanced prompt for better chatbot responses
    prompt_content = """You are a helpful and knowledgeable document assistant chatbot. Your primary role is to help users find information from their documents using an integrated search system.

CORE BEHAVIOR:
- Always try to provide a helpful response, even if the information is partial
- Be conversational and friendly in your tone
- When you have relevant information, present it clearly and confidently
- If information is limited, acknowledge what you do know and offer to help further
- Maintain conversation continuity by referencing previous exchanges when relevant

RESPONSE GUIDELINES:
1. ALWAYS attempt to answer based on available document content
2. If you find relevant information, provide a comprehensive response with specific details
3. If information is partial, say "Based on the available information..." and provide what you can
4. If no relevant information is found, suggest alternative questions or topics the user might explore
5. Never simply say "I don't know" without attempting to be helpful
6. Ask clarifying questions when the user's intent is unclear
7. Provide context and explain technical terms when necessary

CONVERSATION FLOW:
- Acknowledge the user's question
- Search through available documents
- Provide the most relevant information found
- Offer additional help or related information when appropriate
- Reference previous conversation context when it adds value to the current response

Remember: You are designed to be maximally helpful. Even when perfect information isn't available, guide the user toward useful insights or suggest ways to refine their search."""

    # Build conversation context with Cosmos DB history
    chat_prompt = [
        {
            "role": "system",
            "content": f"{prompt_content}"
        }
    ]

    # Get conversation history from Cosmos DB
    cosmos_messages = get_last_messages_from_cosmos(session_id, limit=5)

    if cosmos_messages:
        debug_print(f"Using Cosmos DB conversation history with {len(cosmos_messages)} messages")
        for msg in cosmos_messages:
            chat_prompt.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current user input
    chat_prompt.append({
        "role": "user",
        "content": user_input
    })

    debug_print("Chat Prompt Prepared", {"message_count": len(chat_prompt)})

    last_response = None
    last_error = None

    for attempt in range(max_retries):
        try:
            debug_print(f"Attempt {attempt + 1}/{max_retries}")

            # Add a small delay for subsequent attempts
            if attempt > 0:
                debug_print(f"Waiting {delay} seconds before retry...")
                await asyncio.sleep(delay)

            # Run the completion in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False,
                extra_body={
                    "data_sources": [{
                        "type": "azure_search",
                        "parameters": {
                            "filter": None,
                            "endpoint": f"{search_endpoint}",
                            "index_name": f"{index_name}",
                            "semantic_configuration": "pr1semantic",
                            "authentication": {
                                "type": "api_key",
                                "key": f"{search_key}"
                            },
                            "embedding_dependency": {
                                "type": "endpoint",
                                "endpoint": Embedding_Endpoint,
                                "authentication": {
                                    "type": "api_key",
                                    "key": subscription_key
                                }
                            },
                            "query_type": "semantic",   # Use simple query for better performance
                            "in_scope": True,
                            "strictness": 1,
                            "top_n_documents": 15
                        }
                    }]
                }
            ))

            # Debug: Log the full completion response
            debug_print("Full OpenAI Response", {
                "id": completion.id,
                "model": completion.model,
                "usage": completion.usage.dict() if completion.usage else None,
                "choices_count": len(completion.choices)
            })

            # Extract the main response
            response_content = completion.choices[0].message.content
            debug_print(f"LLM Response Content: {response_content}")

            # Debug: Check if there are any context/citations in the response
            choice = completion.choices[0]
            if hasattr(choice.message, 'context'):
                debug_print("AI Search Context Found", choice.message.context)

            # Check for weak response patterns
            weak_response_phrases = [
                "i don't know", "i do not know", "i don't have information",
                "i cannot find", "i'm not sure", "i don't see", "no information available",
                "i don't have access", "i cannot provide", "i'm unable to", "sorry, i don't have",
                "i don't have any information", "i cannot help", "i'm sorry, but i don't",
                "i don't have specific information", "i cannot locate", "i don't find"
            ]

            response_lower = response_content.lower()
            contains_weak_response = any(phrase in response_lower for phrase in weak_response_phrases)

            # More sophisticated check - only retry if response is both weak AND very short
            is_genuinely_weak = (
                    contains_weak_response and
                    len(response_content.strip()) < 150 and
                    not any(word in response_lower for word in
                            ["however", "although", "but", "based on", "according to", "the document"])
            )

            debug_print(f"Contains weak response phrases: {contains_weak_response}")
            debug_print(f"Is genuinely weak response: {is_genuinely_weak}")
            debug_print(f"Response length: {len(response_content)}")

            # If this is not the last attempt and response is genuinely weak, retry
            if attempt < max_retries - 1 and is_genuinely_weak:
                last_response = response_content
                debug_print("Response is genuinely weak, will retry...")
                continue
            else:
                debug_print("Returning final response")
                return response_content

        except Exception as e:
            last_error = e
            debug_print(f"Error in attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                debug_print("Will retry after error...")
                continue
            else:
                debug_print("Max retries reached, raising error")
                raise e

    # If all retries failed, return the last response or raise the last error
    if last_response:
        debug_print("Returning last response after all retries")
        return last_response
    elif last_error:
        debug_print("Raising last error after all retries")
        raise last_error
    return None
