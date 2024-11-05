# MultiStep Tool Use Reasoning

**Created:** November 3, 2024

## Problem Setting
This project aims to understand and improve how a model performs on a basic MultiStep Tool Use Reasoning task, focusing on basic math word problems and conditionals. The key constraint is that the model should not compute the answer directly; instead, it should construct the appropriate mathematical expression.

### Canonical Example
**Problem:**  
*Alice has 15 books, and she plans to buy 3 packs of books with 5 books in each pack. If the total number of books exceeds 30 after buying, she will donate 10 books. Otherwise, she will keep them all. How many books will Alice have in the end?*

**Expected Response:**  
- Calculate `result1 = (15 + 3 * 5)`
- If `(result1 < 30)`: return `(15 + 3 * 5)`
- Else: return `(15 + 3 * 5 - 10)`

## Model Specifications
- **Base Model:** `gpt-3.5-turbo`

## Method
The following steps outline the approach for achieving the desired model behavior:

- **Generate a Supervised Fine-Tuning (SFT) Dataset**: Using a larger model (`gpt-4`).
- **Fine-Tune**: Apply fine-tuning to the base model.
- **Evaluate**: Assess performance to verify if the model produces expected reasoning steps.

## Results

| Model                | Accuracy | Description                                             |
|----------------------|----------|---------------------------------------------------------|
| base                 | 0.00%    | Base `gpt-3.5-turbo` model without any additional context |
| base_with_context    | 0.00%    | Base model with 1 example in the system prompt          |
| sft-10samples        | 85.00%   | Fine-tuned model with 10 sample training examples       |
| sft-30samples        | 90.00%   | Fine-tuned model with 30 sample training examples       |

## Demo
Initial results demonstrate that supervised fine-tuning (SFT) helps achieve the intended behavior. Key insights include:

- **Improved Stability**: Using SFT enables consistent model responses, addressing limitations found when relying solely on prompts.
- **Scalability**: Hard-coding specific instructions in the context proved cumbersome and unscalable. SFT allows this capability to integrate smoothly as part of a broader functionality set without requiring rigid instructions.

[Math Tool Use Reasoning - Demo](https://github.com/user-attachments/assets/6c5c5190-54f2-45f3-ab13-2294ac8ffaea)

