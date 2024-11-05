import json
import agent
import calculator
from absl import app, flags

# Define the flags
FLAGS = flags.FLAGS
flags.DEFINE_string("eval_dataset", "eval.json", "Path to the evaluation dataset JSON file.")

# Dictionary to map labels to actual model names
model_map = {
    "base": "gpt-3.5-turbo",
    "base_with_context": "gpt-3.5-turbo",
    "sft-10samples": "ft:gpt-3.5-turbo-0125:personal:math-multi-finetuned-v1:APlkvVep",
    "sft-30samples": "ft:gpt-3.5-turbo-0125:personal:math-multi-finetuned-v1:APmTgIOS",
}

# Define the in-context example
in_context_example = (
    "Example:\n"
    "User: Sarah has 10 plants and plans to buy 4 packs of plants with 6 plants in each pack. "
    "If the total number of plants exceeds 35 after buying, she will gift 5 plants. Otherwise, she will keep them all.\n"
    "Assistant's tool calls:\n"
    "1. Tool call:\n"
    "   - id: call_id_1\n"
    "   - type: function\n"
    "   - function:\n"
    "       name: calculate\n"
    "       arguments: {\"expression\": \"10 + 4 * 6\"}\n"
    "2. Tool call:\n"
    "   - id: call_id_2\n"
    "   - type: function\n"
    "   - function:\n"
    "       name: calculate\n"
    "       arguments: {\"expression\": \"10 + 4 * 6 - 5\"}\n"
    "3. Tool call:\n"
    "   - id: call_id_3\n"
    "   - type: function\n"
    "   - function:\n"
    "       name: calculate\n"
    "       arguments: {\"expression\": \"10 + 4 * 6\"}\n\n"
    "Now, given a similar problem, follow the same approach to respond."
)


def evaluate_model(model_label, eval_data, use_context=False):
    """
    Evaluates the performance of a specified model on a dataset of math-related prompts.

    Args:
        model_label (str): The label for the model to be evaluated (e.g., 'base', 'sft-10samples').
        eval_data (list): The evaluation dataset, where each example includes 'user_input' (prompt)
                          and 'expected_tool_calls' (expected output function calls).
        use_context (bool): If True, includes the in-context example in the prompt to the model.

    Outputs:
        Prints the accuracy of the model based on the correct tool calls.
    """
    
    # Retrieve the actual model name from the model map
    model = model_map[model_label]

    # Initialize counters for correct and total expected tool calls
    correct_tool_calls = 0
    total_expected_tool_calls = 0

    # Run evaluation for each example in the dataset
    # Inside the evaluate_model function
    for example in eval_data:
        # Prepare the system message with instructions
        system_content = (
            "You are a helpful Math tutor. "
            "Your task is to ALWAYS use the calculator tool for any mathematical calculations. "
            "When presented with a problem, condense it into a mathematical expression and pass that to the calculator. "
            "Do not perform calculations directly yourself. "
            "You may use multiple calls to the calculator as needed. "
            "Note: The only operations available to you are addition, subtraction, multiplication, division, and brackets."
        )

        # If using context, prepend the in-context example to the system message
        if use_context:
            system_content = in_context_example + "\n\n" + system_content

        # Construct the prompt with the (optional) in-context example in the system message
        prompt = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": example["user_input"]
            }
        ]

        # Get model response
        response = agent.chat_completion_request(
            model=model,
            messages=prompt,
            functions=calculator.funcs()
        )

        # Extract the assistant's function calls from the response
        actual_tool_calls = response["choices"][0]["message"]["tool_calls"]

        # Retrieve expected tool calls from the dataset and set up comparison limit
        expected_tool_calls = example["expected_tool_calls"]
        num_tool_calls_to_assert = min(len(actual_tool_calls), len(expected_tool_calls))

        # Compare each actual tool call to the corresponding expected tool call
        for i in range(num_tool_calls_to_assert):
            actual, expected = actual_tool_calls[i], expected_tool_calls[i]
            actual_function = actual["function"]
            expected_function = expected  # Assuming `expected` is also a dictionary here
            
            # Initialize flag for checking if the current function call matches
            is_call_correct = True

            # Compare each key-value pair in expected with actual, ignoring whitespace differences
            for key, expected_value in expected_function.items():
                actual_value = actual_function.get(key)
                
                # Strip whitespace if values are strings for more reliable comparison
                if isinstance(actual_value, str):
                    actual_value = actual_value.strip()
                if isinstance(expected_value, str):
                    expected_value = expected_value.strip()
                
                if actual_value != expected_value:
                    is_call_correct = False
                    break
            
            # If all key-value pairs match, count this function call as correct
            if is_call_correct:
                correct_tool_calls += 1

        # Update the total expected tool calls count
        total_expected_tool_calls += len(expected_tool_calls)

    # Calculate and report accuracy
    accuracy = correct_tool_calls / total_expected_tool_calls * 100 if total_expected_tool_calls else 0
    print(f"Accuracy for model '{model_label}': {accuracy:.2f}%")

def main(argv):
    """
    Main function that loads the evaluation dataset and evaluates all models in the model_map.

    Args:
        argv (list): Command-line arguments (not used here).
    """
    
    # Load the dataset from the path provided in the flag
    with open(FLAGS.eval_dataset) as f:
        eval_data = json.load(f)

    # Evaluate each model in the model map
    for label in model_map:
        # If model type requires in-context example, set use_context to True
        use_context = "with_context" in label
        evaluate_model(label, eval_data, use_context=use_context)

# Entry point for the script
if __name__ == "__main__":
    app.run(main)
