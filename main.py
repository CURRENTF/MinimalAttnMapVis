import torch
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt
import os
from llama import GetAttnMapLM


def load_model_and_tokenizer(model_path):
    model = GetAttnMapLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tkn = AutoTokenizer.from_pretrained(model_path)
    return model, tkn


def greedy_decode(input_text, model, tkn, max_new_tokens=50):
    """
    Generates text using greedy decoding and collects attention scores from decode phase.

    Args:
        input_text (str): The prompt text.
        model: The GetAttnMapLM model instance.
        tkn: The tokenizer instance.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        tuple: (output_text, collected_decode_attentions)
               output_text (str): The generated text including the prompt.
               collected_decode_attentions (list): A list of tuples. Each tuple contains
                                                   attention tensors for each layer for a single
                                                   decode step. Attention tensor shape is
                                                   (batch_size, num_heads, 1, key_sequence_length).
                                                   The list corresponds to the generated tokens
                                                   *after* the first one derived from the prompt.
    """
    model.eval()  # Set model to evaluation mode

    inputs = tkn(input_text, return_tensors="pt").to(model.device)
    input_ids_prompt = inputs.input_ids

    current_generated_ids = input_ids_prompt  # Holds the full sequence including prompt and generated tokens

    collected_decode_attentions = []

    past_key_values = None

    if max_new_tokens == 0:
        output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
        return output_text, []

    with torch.no_grad():  # Disable gradient calculations for inference
        # --- Prefill Phase ---
        # Process the entire prompt to get initial past_key_values and logits for the first token.
        # GetAttnMapLM is specified to NOT output attentions in this prefill phase.
        prefill_outputs = model(
            input_ids=input_ids_prompt,
            use_cache=True
            # output_attentions is internally False for prefill by GetAttnMapLM
            # or we could explicitly pass output_attentions=False if GetAttnMapLM
            # respects user overrides for its special behavior.
            # Assuming GetAttnMapLM handles this correctly based on phase.
        )
        logits_prefill = prefill_outputs.logits
        past_key_values = prefill_outputs.past_key_values

        # Generate the first token based on the prompt
        next_token_logits = logits_prefill[:, -1, :]  # Logits for the last token of the prompt
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

        # If EOS is generated after the first token, finish early.
        # No decode attentions collected yet as this token came from prefill_outputs.
        if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
            output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
            return output_text, collected_decode_attentions  # Should be empty

        # --- Decode Phase ---
        # Loop to generate remaining (max_new_tokens - 1) tokens, as one is already generated.
        for _ in range(max_new_tokens - 1):
            # The input for this decode step is the token generated in the previous step.
            decode_input_ids = next_token_id

            outputs_decode_step = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True
                # output_attentions is internally True for decode by GetAttnMapLM
            )

            logits_decode_step = outputs_decode_step.logits
            past_key_values = outputs_decode_step.past_key_values  # Update KV cache

            # Collect attention scores from this decode step
            # GetAttnMapLM is expected to provide 'attentions' in outputs_decode_step.
            if hasattr(outputs_decode_step, 'attentions') and outputs_decode_step.attentions is not None:
                # attentions is a tuple (one for each layer) of torch.FloatTensor
                # of shape (batch_size, num_heads, sequence_length=1, key_sequence_length)
                # Detach and move to CPU to save GPU memory, especially for long generations.
                step_attentions = tuple(att.detach().cpu() for att in outputs_decode_step.attentions)
                collected_decode_attentions.append(step_attentions)
            # If GetAttnMapLM guarantees attentions in decode, no 'else' needed.
            # If not, one might log a warning or append a placeholder.

            # Determine the next token
            next_token_logits = logits_decode_step[:, -1, :]  # Logits for the newly generated token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token to the sequence
            current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

            # Check for EOS token
            if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
                break

    output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
    return output_text, collected_decode_attentions


def load_data(data_path, sample_num=10):
    """
    Loads data from a JSONL file.

    Args:
        data_path (str): Path to the JSONL file.
        sample_num (int): Number of samples to load from the beginning of the file.

    Returns:
        list: A list of dictionaries, where each dictionary is a loaded JSON object.
    """
    data_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num:
                    break
                d = json.loads(line.strip())
                data_list.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {data_path} on line {i + 1}: {e}")
        # Optionally, continue loading other lines or return partially loaded data
        return data_list
    return data_list


def cumsum_attn_score_for_each_layer(attn_map_at_one_decode_step):
    """
    Processes attention scores for each layer from a single decode step.
    It averages over heads and then calculates the cumulative sum over the key sequence length.

    Args:
        attn_map_at_one_decode_step (tuple): A tuple of attention tensors from a single decode step.
                                            Each tensor corresponds to a layer and has a shape like
                                            (batch_size, num_heads, query_length=1, key_sequence_length).
                                            For typical decode steps, batch_size=1 and query_length=1.

    Returns:
        list: A list of processed attention tensors. Each tensor in the list has shape
              (batch_size, key_sequence_length) after averaging over heads and
              (batch_size, key_sequence_length) after cumsum.
    """
    processed_layers_attn = []
    if attn_map_at_one_decode_step is None:
        return []

    for layer_attn in attn_map_at_one_decode_step:
        # layer_attn expected shape: (batch_size, num_heads, 1, key_len)

        # Ensure it's 4D, though typical decode attention might be (batch, heads, 1, key_len)
        if layer_attn.ndim != 4:
            raise ValueError(f"Expected 4D attention tensor, got {layer_attn.ndim}D with shape {layer_attn.shape}")

        # Squeeze the query_length dimension (dim 2), assuming it's always 1 in decode.
        # If query_length > 1, this logic would need adjustment (e.g., mean over query_length too, or process each query pos).
        if layer_attn.shape[2] == 1:
            squeezed_attn = layer_attn.squeeze(2)  # Shape: (batch_size, num_heads, key_len)
        else:
            raise ValueError(f"attn shape: {layer_attn.shape}")

        # 1. Average over the head dimension (dim 1 of squeezed_attn)
        avg_head_attn = torch.mean(squeezed_attn, dim=1)  # Shape: (batch_size, key_len)
        # 2. Calculate cumulative sum along the key_sequence_length dimension (last dimension)
        cumsum_attn = torch.cumsum(avg_head_attn, dim=-1)  # Shape: (batch_size, key_len)
        processed_layers_attn.append(cumsum_attn)

    return processed_layers_attn


def vis_cumsum_attn_for_each_layer_each_step(step_i, layer_i, attn_cumsum_tensor, output_dir="visualizations"):
    """
    Visualizes the cumulative attention scores for a given layer and decode step,
    and saves it as a JPG file.

    Args:
        step_i (int): The current decode step index (0-based).
        layer_i (int): The current layer index (0-based).
        attn_cumsum_tensor (torch.Tensor): The cumulative attention tensor for this layer/step.
                                           Expected shape (batch_size, key_sequence_length).
                                           Typically batch_size is 1.
        output_dir (str): Directory to save the visualization.
    """
    if attn_cumsum_tensor is None or attn_cumsum_tensor.numel() == 0:
        print(f"  Visualization skipped for step {step_i + 1}, layer {layer_i + 1}: Empty tensor.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}. Visualizations will not be saved.")
            return

    # Assuming batch_size is 1, so we take the first element.
    # If batch_size > 1, one might loop or plot all, or average.
    # For this specific function, we'll plot the first batch element.
    if attn_cumsum_tensor.shape[0] > 1:
        print(
            f"  Warning: attn_cumsum_tensor has batch size {attn_cumsum_tensor.shape[0]} for step {step_i + 1}, layer {layer_i + 1}. Visualizing first batch element only.")

    data_to_plot = attn_cumsum_tensor[0].cpu().numpy()  # Shape: (key_sequence_length,)

    if data_to_plot.ndim != 1:
        print(
            f"  Visualization skipped for step {step_i + 1}, layer {layer_i + 1}: Expected 1D data after batch selection, got {data_to_plot.ndim}D.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(data_to_plot)
    plt.title(f"Cumulative Attention: Decode Step {step_i + 1}, Layer {layer_i + 1}")
    plt.xlabel("Key Sequence Position (Token Index in KV Cache)")
    plt.ylabel("Cumulative Attention Score (Averaged over Heads)")
    plt.grid(True)

    filename = f"cumsum_step_{step_i + 1}_layer_{layer_i + 1}.jpg"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath)
        print(f"    Saved visualization to {filepath}")
    except Exception as e:
        print(f"    Error saving plot to {filepath}: {e}")
    plt.close()  # Close the figure to free memory


if __name__ == '__main__':
    MODEL_PATH = "path/to/your/GetAttnMapLM/model"
    DATA_PATH = "path/to/your/data.jsonl"
    SAMPLE_NUM = 2
    MAX_NEW_TOKENS_GENERATION = 20

    # --- Placeholder for running the script ---
    print("Starting example usage (ensure MODEL_PATH and DATA_PATH are set)...")

    try:
        # 1. Load model and tokenizer
        print(f"Loading model and tokenizer from: {MODEL_PATH}")
        model, tkn = load_model_and_tokenizer(MODEL_PATH)

        # 2. Load data
        print(f"Loading data from: {DATA_PATH} (first {SAMPLE_NUM} samples)")
        data_samples = load_data(DATA_PATH, SAMPLE_NUM)
        if not data_samples:
            print("No data loaded. Exiting.")
        else:
            print(f"Loaded {len(data_samples)} samples.")

        for i, sample in enumerate(data_samples):
            input_text = sample

            print(f"Input text: {input_text}\n")

            # 3. Greedy decode and get attention scores
            output_text, collected_attentions = greedy_decode(input_text, model, tkn, max_new_tokens=MAX_NEW_TOKENS_GENERATION)
            output_text = input_text + output_text

            print(f"Generated text: {output_text}")
            print(f"Number of decode steps for which attentions were collected: {len(collected_attentions)}")

            # 4. Process attention scores for each step (if any were collected)
            all_steps_processed_attns = []
            if collected_attentions:
                for step_idx, attn_map_one_step in enumerate(collected_attentions):
                    print(f"  Processing attentions for decode step {step_idx + 1}:")
                    # attn_map_one_step is a tuple of layer attentions for this decode step
                    # Each layer_attn tensor is (batch_size, num_heads, 1, key_len_at_this_step)
                    processed_layer_attns_for_step = cumsum_attn_score_for_each_layer(attn_map_one_step)
                    all_steps_processed_attns.append(processed_layer_attns_for_step)

                    # Visualize for this step
                    for layer_idx, cumsum_attn_tensor in enumerate(processed_layer_attns_for_step):
                        print(f"    Layer {layer_idx + 1} processed cumsum attention shape: {cumsum_attn_tensor.shape}")
                        vis_cumsum_attn_for_each_layer_each_step(
                            step_idx, layer_idx, cumsum_attn_tensor
                        )
            else:
                print("No attention scores were collected (e.g., max_new_tokens was small or EOS met early).")
            # `all_steps_processed_attns` is now a list (decode steps) of lists (layers)
            # where each inner element is a tensor of shape (batch_size, key_len_at_that_step)

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure all required libraries (like 'llama') are installed.")
    except Exception as e:
        # Catch other potential errors, e.g. during model loading if paths are wrong.
        print(f"An unexpected error occurred: {e}")
