import agent
import streamlit as st 
import calculator

st.title('MultiStep Mathematical Reasoning')

# Dictionary to map labels to actual model names
model_map = {
	"base": "gpt-3.5-turbo",
	"sft-10samples": "ft:gpt-3.5-turbo-0125:personal:math-multi-finetuned-v1:APlkvVep",
	"sft-30samples": "ft:gpt-3.5-turbo-0125:personal:math-multi-finetuned-v1:APmTgIOS"
}

# Dropdowns for selecting two models with labels instead of actual model names
model_1_label = st.selectbox("Select Model 1", list(model_map.keys()), index=0)
model_2_label = st.selectbox("Select Model 2", list(model_map.keys()), index=1)

# Retrieve the actual model names based on selected labels
model_1 = model_map[model_1_label]
model_2 = model_map[model_2_label]

# Text input for query
query = st.text_input("Query")

if query:
	messages = [
		{
			"role": "system",
			"content": "You are a helpful Math tutor. \
			Your task is to ALWAYS use the calculator tool for any mathematical calculations. \
			When presented with a problem, condense it into a mathematical expression and pass that to the calculator. \
			Do not perform calculations directly yourself. \
			You may use multiple calls to the calculator as needed. \
			Note: The only operations available to you are addition, subtraction, multiplication, division, and brackets."
		},
		{
			"role": "user",
			"content": query
		}
	]

	# Get responses from both models
	response_1 = agent.chat_completion_request(
		model=model_1,
		messages=messages,
		functions=calculator.funcs()
	)
	response_2 = agent.chat_completion_request(
		model=model_2,
		messages=messages,
		functions=calculator.funcs()
	)

	# Extract responses
	assistant_message_1 = response_1["choices"][0]["message"]
	assistant_message_2 = response_2["choices"][0]["message"]

	# Display responses side by side
	col1, col2 = st.columns(2)
	with col1:
		st.subheader(f"Response from Model 1: {model_1_label}")
		st.write(assistant_message_1)
	with col2:
		st.subheader(f"Response from Model 2: {model_2_label}")
		st.write(assistant_message_2)
