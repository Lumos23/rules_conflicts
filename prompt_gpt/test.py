from prompt_gpt import initialize, get_response_with_retries, DEFAULT_MODEL

initialize(DEFAULT_MODEL, verbose=True)
response = get_response_with_retries(
    directive="You are a helpful AI assistant that can add two numbers.",
    prompt="What is the value of 265 + 43?"
)

print(response)

# assert 0, "A test exception."