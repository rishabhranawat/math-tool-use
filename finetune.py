import os
import openai
from absl import app, flags

# Define command-line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Path to the input dataset file (.jsonl) for fine-tuning')
flags.DEFINE_string('suffix', 'math-multi-finetuned-v1', 'Suffix for the fine-tuned model')

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def upload_file(dataset_path):
    """
    Uploads a file for fine-tuning.
    
    Args:
        dataset_path (str): Path to the dataset file to be uploaded.
    
    Returns:
        str: The ID of the uploaded file.
    """
    with open(dataset_path, "rb") as f:
        response = openai.File.create(file=f, purpose="fine-tune")
    print("File upload response:", response)
    return response.id

def create_fine_tuning_job(file_id, suffix):
    """
    Creates a fine-tuning job using the uploaded file ID.
    
    Args:
        file_id (str): The ID of the file to use for fine-tuning.
        suffix (str): The suffix for the fine-tuned model.

    Returns:
        dict: The response from the fine-tuning job creation.
    """
    response = openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo-0125",
        suffix=suffix
    )
    print("Fine-tuning job response:", response)
    return response

def main(argv):
    # Ensure the required flags are provided
    if not FLAGS.dataset:
        raise ValueError("You must specify the dataset file path using --dataset flag.")
    
    # Step 1: Upload the dataset file
    file_id = upload_file(FLAGS.dataset)
    
    # Step 2: Create the fine-tuning job with the uploaded file ID and suffix
    create_fine_tuning_job(file_id, FLAGS.suffix)

if __name__ == "__main__":
    flags.mark_flag_as_required('dataset')
    app.run(main)
