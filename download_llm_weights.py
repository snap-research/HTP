import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    """
    Prompts the user for a Hugging Face model name and downloads the model
    and its tokenizer to a local directory.
    """
    # 1. Prompt the user for the model name
    model_name = input("Enter the Hugging Face model name (e.g., mistralai/Mistral-7B-Instruct-v0.3): ")

    # Basic validation for the input
    if not model_name or '/' not in model_name:
        print("❌ Invalid input. Please enter a valid model name in the 'organization/model-name' format.")
        return

    try:
        # 2. Define the local directory to save the model
        # Takes the part after the '/' (e.g., 'Mistral-7B-Instruct-v0.3')
        local_name = model_name.split('/')[-1]
        local_dir = f"./weights/{local_name}"

        # Create the target directory if it doesn't already exist
        os.makedirs(local_dir, exist_ok=True)
        print(f"\n📂 Saving model to: {local_dir}")

        # 3. Download and save the tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        print("Tokenizer downloaded.")

        # 4. Download and save the model
        print("Downloading model... This may take a significant amount of time and disk space.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        print("Model downloaded.")

        # 5. Print success message
        print(f"\n✅ Success! Model and tokenizer for '{model_name}' are saved in '{local_dir}'")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the model name and your internet connection.")


# Standard Python entry point
if __name__ == "__main__":
    main()