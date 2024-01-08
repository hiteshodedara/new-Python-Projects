import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained "phi-2" model and tokenizer
model_name = "phi-2"  # Replace with the actual name of the "phi-2" model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response using "phi-2"
def generate_phi2_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Function to interact with the "phi-2"-powered chatbot
def interact_with_phi2_chatbot():
    print("Hello! I'm your custom chatbot. You can start interacting. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Custom Chatbot: Goodbye!")
            break
        response = generate_phi2_response(user_input)
        print("Custom Chatbot:", response)

# Start the interaction
interact_with_phi2_chatbot()
