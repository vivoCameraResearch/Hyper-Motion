mport os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from the specified path"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to("cuda")
    return tokenizer, model

def process_text_with_model(text, tokenizer, model):
    """Process text with the model to rewrite and streamline it"""
    prompt = f"""Rewrite the following text , Do not start with something similar to 'The video/scene/frame shows' or "In this video/scene/frame". Remove the subjective content deviates from describing the visual content of the video. For instance, a sentence like "It gives a feeling of ease and tranquility and makes people feel comfortable" is considered subjective. Remove the non-existent description that does not in the visual content of the video, For instance, a sentence like "There is no visible detail that could be used to identify the individual beyond what is shown." is considered as the non-existent description. Please focus on retaining the action descriptions of the characters in the text, but otherwise streamline the text, trying to keep it to no more than 50 words, and if the text has more than one paragraph, please turn it into one paragraph. :\n\n{text}\n\nResult:"""

    # Encode the text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
   
    # Generate results
    outputs = model.generate(**inputs, max_length=2048, num_return_sequences=1)
   
    # Decode the generated text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
   
    # Extract the result part
    if "Result:" in result:
        result = result.split("Result:")[-1].strip()

    rewritten_lines = result.split("\n")
    rewritten_text = rewritten_lines[0].strip() if rewritten_lines else result.strip()
   
    return rewritten_text

def batch_process_files(input_folder, output_folder, tokenizer, model):
    """Process all text files in the input folder and save results to the output folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
           
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
           
            processed_content = process_text_with_model(content, tokenizer, model)
           
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
           
            print(f"Processed: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Process text files using a language model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input text files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the processed text files")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use (default: 0)")
   
    args = parser.parse_args()
   
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
    print(f"Loading model from {args.model_path}...")
    tokenizer, model = load_model_and_tokenizer(args.model_path)
   
    print(f"Processing files from {args.input_folder} to {args.output_folder}...")
    batch_process_files(args.input_folder, args.output_folder, tokenizer, model)
   
    print("All files processed successfully!")

if __name__ == "__main__":
    main()

