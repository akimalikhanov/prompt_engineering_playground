from vllm import LLM

# Load the model (it will download the model if not already cached)
model = LLM(model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# Prepare the prompt
prompt = "Tell me shortly: why did Trump won in 2016?"

# Run the model to get predictions
output = model.generate([prompt])

# Print the output
print("Response:", output[0]['text'])
