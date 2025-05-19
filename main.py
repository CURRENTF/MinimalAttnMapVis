import torch
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt
import os
# Assuming 'llama.py' with GetAttnMapLM is in the same directory or PYTHONPATH
from llama import GetAttnMapLM

# --- Configuration ---
MODEL_PATH = "../models/Llama-3-8B-Instruct-262k"  # Update with your model path
DATA_PATH = "../OmniKV/benchmark/long_bench/data/hotpotqa.jsonl"  # Update with your data path
SAMPLE_NUM = 1  # Number of samples to process from the data file
MAX_NEW_TOKENS_GENERATION = 20  # Max new tokens (after the first one derived from prompt)
OUTPUT_VIS_DIR = "visualizations"  # Directory to save plots
SIMILARITY_TOP_K = 1024  # K for top-k tokens in attention similarity calculation
PLOT_DPI = 300  # DPI for saved plots

# Ensure visualization directory exists
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


# --- Model and Tokenizer Loading ---
def load_model_and_tokenizer(model_path):
    """Loads the GetAttnMapLM model and its tokenizer."""
    print(f"Loading model and tokenizer from: {model_path}")
    model = GetAttnMapLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tkn = AutoTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded.")
    return model, tkn


# --- Data Loading ---
def load_data(data_path, sample_num=10):
    """Loads data from a JSONL file."""
    data_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num:
                    break
                d = json.loads(line.strip())
                # Formatting the input as context and question
                data_list.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
        print(f"Loaded {len(data_list)} samples from {data_path}.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {data_path}: {e}")
    return data_list


# --- Greedy Decoding with Attention Collection ---
def greedy_decode(input_text, model, tkn, max_new_tokens=50):
    """
    Generates text using greedy decoding and collects attention scores from decode phase.
    Args:
        input_text (str): The prompt text.
        model: The GetAttnMapLM model instance.
        tkn: The tokenizer instance.
        max_new_tokens (int): Max new tokens to generate. Attentions are collected for tokens
                              generated *after* the first one (which is derived from prefill).
    Returns:
        tuple: (output_text, collected_decode_attentions)
               output_text (str): Generated text including prompt.
               collected_decode_attentions (list): List of tuples (one per decode step).
                                                   Each tuple contains layer attentions:
                                                   (batch_size, num_heads, 1, key_sequence_length).
    """
    model.eval()
    inputs = tkn(input_text, return_tensors="pt").to(model.device)
    input_ids_prompt = inputs.input_ids
    current_generated_ids = input_ids_prompt
    collected_decode_attentions = []
    past_key_values = None

    if max_new_tokens == 0:  # Generate 0 new tokens, just return prompt
        return tkn.decode(current_generated_ids[0], skip_special_tokens=True), []

    with torch.no_grad():
        # --- Prefill Phase ---
        # GetAttnMapLM is specified to NOT output attentions in this prefill phase.
        prefill_outputs = model(input_ids=input_ids_prompt, use_cache=True)
        logits_prefill = prefill_outputs.logits
        past_key_values = prefill_outputs.past_key_values

        # Generate the first token based on the prompt
        next_token_logits = logits_prefill[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

        if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
            return tkn.decode(current_generated_ids[0], skip_special_tokens=True), []

        # --- Decode Phase ---
        # Loop for remaining (max_new_tokens - 1) tokens, as one is already generated.
        for _ in range(max_new_tokens - 1):
            decode_input_ids = next_token_id  # Input for this step is the previously generated token
            outputs_decode_step = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True  # GetAttnMapLM should handle output_attentions=True for decode phase
            )

            logits_decode_step = outputs_decode_step.logits
            past_key_values = outputs_decode_step.past_key_values

            # Collect attention scores from this decode step
            if hasattr(outputs_decode_step, 'attentions') and outputs_decode_step.attentions is not None:
                # Detach and move to CPU to save GPU memory
                step_attentions = tuple(att.detach().cpu() for att in outputs_decode_step.attentions)
                collected_decode_attentions.append(step_attentions)

            next_token_logits = logits_decode_step[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

            if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
                break

    output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
    return output_text, collected_decode_attentions


# --- Attention Processing ---
def cumsum_attn_score_for_each_layer(attn_map_at_one_decode_step):
    """
    Processes layer attentions from a single decode step: averages over heads, then cumsum over key sequence.
    Args:
        attn_map_at_one_decode_step (tuple): Tuple of attention tensors from a single decode step.
                                            Each tensor (layer_attn) has shape (B, H, 1, K_len).
    Returns:
        list: A list of processed attention tensors. Each tensor in the list has shape (B, K_len).
    """
    processed_layers_attn = []
    if not attn_map_at_one_decode_step: return []

    for layer_attn in attn_map_at_one_decode_step:
        if layer_attn.ndim != 4 or layer_attn.shape[2] != 1:
            # This might indicate an issue with GetAttnMapLM or unexpected input
            # For now, we'll assume correct shapes as per GetAttnMapLM's design for decode steps.
            # If robust handling is needed, one might append a placeholder or log a more severe warning.
            print(f"Warning: Unexpected attention tensor shape {layer_attn.shape}. Expected (B, H, 1, K).")
            continue  # Skip this layer's processing for this step if shape is wrong

        squeezed_attn = layer_attn.squeeze(2)  # Shape: (batch_size, num_heads, key_len)
        avg_head_attn = torch.mean(squeezed_attn, dim=1)  # Shape: (batch_size, key_len)
        cumsum_attn = torch.cumsum(avg_head_attn, dim=-1)  # Shape: (batch_size, key_len)
        processed_layers_attn.append(cumsum_attn)
    return processed_layers_attn


def cal_attn_map_similarity(raw_layer_attn_i, raw_layer_attn_j, top_k_tokens):
    """
    Calculates a similarity score between two layers' attention patterns for a single decode step.
    Similarity is defined by how much layer j focuses on the top-k tokens of layer i.
    Args:
        raw_layer_attn_i (torch.Tensor): Raw attention tensor for layer i.
                                         Shape (batch_size, num_heads, 1, key_sequence_length).
        raw_layer_attn_j (torch.Tensor): Raw attention tensor for layer j.
                                         Shape (batch_size, num_heads, 1, key_sequence_length).
        top_k_tokens (int): Number of top tokens from layer i's attention to consider.
    Returns:
        float: The similarity score.
    """
    # Ensure tensors are not empty (e.g., key_sequence_length is 0)
    if raw_layer_attn_i.shape[-1] == 0 or raw_layer_attn_j.shape[-1] == 0:
        return 0.0

    # Assuming batch_size = 1 for decode steps.
    # Squeeze query_length dim (dim 2), then average over heads (dim 1 of squeezed tensor).
    avg_head_attn_i = torch.mean(raw_layer_attn_i.squeeze(2), dim=1).squeeze(0)  # Shape: (key_len,)
    avg_head_attn_j = torch.mean(raw_layer_attn_j.squeeze(2), dim=1).squeeze(0)  # Shape: (key_len,)

    # Check if averaging resulted in 1D tensors and key_len > 0
    if not (avg_head_attn_i.ndim == 1 and avg_head_attn_j.ndim == 1 and avg_head_attn_i.shape[0] > 0):
        return 0.0

    actual_k = min(top_k_tokens, avg_head_attn_i.shape[0])
    if actual_k == 0:  # If key_len is 0 (already checked) or top_k_tokens is 0
        return 0.0

    _, top_indices_i = torch.topk(avg_head_attn_i, k=actual_k, dim=-1)

    # Sum the attention scores of layer j at the top-k positions identified by layer i
    similarity_score = torch.sum(avg_head_attn_j[top_indices_i]).item()
    return similarity_score


# --- Visualization ---
def vis_cumsum_attn_for_each_layer_each_step(sample_id, step_i, layer_i, attn_cumsum_tensor, output_dir, dpi):
    """Visualizes cumulative attention for a layer/step."""
    if attn_cumsum_tensor is None or attn_cumsum_tensor.numel() == 0: return

    # Assuming batch_size is 1, take the first element.
    data_to_plot = attn_cumsum_tensor[0].cpu().float().numpy()  # Shape: (key_sequence_length,)
    if data_to_plot.ndim != 1: return  # Should be 1D after batch selection

    plt.figure(figsize=(10, 4))
    plt.plot(data_to_plot)
    plt.title(f"Sample {sample_id + 1} - CumSum Attn: Decode Step {step_i + 1}, Layer {layer_i + 1}")
    plt.xlabel("Key Sequence Position (Token Index in KV Cache)")
    plt.ylabel("Cumulative Attention Score (Avg. Heads)")
    plt.grid(True)

    filename = f"sample_{sample_id + 1}_cumsum_step_{step_i + 1}_layer_{layer_i + 1}.jpg"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=dpi)
    except Exception as e:
        print(f"    Error saving cumsum plot to {filepath}: {e}")
    finally:
        plt.close()


def vis_layer_similarity_matrix(sample_id, step_i, similarity_matrix, output_dir, dpi):
    """Visualizes the layer-wise attention similarity matrix as a heatmap."""
    plt.figure(figsize=(12, 10))
    # Using aspect='auto' and letting imshow determine vmin/vmax, or they can be fixed.
    plt.imshow(similarity_matrix.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label="Attention Similarity Score")
    plt.title(f"Sample {sample_id + 1} - Layer Attention Similarity: Decode Step {step_i + 1}")
    plt.xlabel("Layer Index (j)")
    plt.ylabel("Layer Index (i)")

    num_layers = similarity_matrix.shape[0]
    # Adjust ticks for readability based on number of layers
    if num_layers > 0:
        ticks = list(range(num_layers))
        tick_labels = [str(t + 1) for t in ticks]  # 1-indexed labels
        if num_layers <= 32:  # Show all ticks for fewer layers
            plt.xticks(ticks, tick_labels)
            plt.yticks(ticks, tick_labels)
        else:  # Show sparse ticks for many layers to avoid clutter
            step_size = num_layers // 10 or 1  # Ensure step_size is at least 1
            sparse_ticks = ticks[::step_size]
            sparse_tick_labels = tick_labels[::step_size]
            plt.xticks(sparse_ticks, sparse_tick_labels)
            plt.yticks(sparse_ticks, sparse_tick_labels)

    filename = f"sample_{sample_id + 1}_layer_similarity_step_{step_i + 1}.jpg"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=dpi)
        print(f"    Saved layer similarity heatmap for step {step_i + 1} to {filepath}")
    except Exception as e:
        print(f"    Error saving layer similarity plot to {filepath}: {e}")
    finally:
        plt.close()


# --- Main Execution ---
if __name__ == '__main__':
    model, tkn = load_model_and_tokenizer(MODEL_PATH)
    num_model_layers = model.config.num_hidden_layers  # Get total number of layers from model config
    print(f"Model has {num_model_layers} hidden layers.")

    data_samples = load_data(DATA_PATH, SAMPLE_NUM)

    if not data_samples:
        print("No data loaded or error in loading. Exiting.")
    else:
        for sample_idx, input_text in enumerate(data_samples):
            print(f"\n--- Processing Sample {sample_idx + 1}/{len(data_samples)} ---")
            # print(f"Input text (first 200 chars): {input_text[:200]}...")

            output_text, collected_attentions = greedy_decode(
                input_text, model, tkn, max_new_tokens=MAX_NEW_TOKENS_GENERATION
            )

            # print(f"Generated text (first 200 chars): {output_text[:200]}...") # Optional: print generated text
            print(f"Collected attentions for {len(collected_attentions)} decode steps.")

            if not collected_attentions:
                print(
                    "No decode attentions collected for this sample (e.g., max_new_tokens too small or EOS met early).")
                continue

            for step_idx, attn_map_one_step in enumerate(collected_attentions):
                # attn_map_one_step is a tuple of layer attentions for this decode step.
                # Each layer_attn tensor is (batch_size, num_heads, 1, key_len_at_this_step).
                print(f"  Processing & Visualizing Attentions for Decode Step {step_idx + 1}:")

                # 1. Cumulative Attention Visualization (per layer)
                processed_layer_attns_for_step = cumsum_attn_score_for_each_layer(attn_map_one_step)
                for layer_idx, cumsum_attn_tensor in enumerate(processed_layer_attns_for_step):
                    vis_cumsum_attn_for_each_layer_each_step(
                        sample_idx, step_idx, layer_idx, cumsum_attn_tensor,
                        OUTPUT_VIS_DIR, PLOT_DPI
                    )

                # 2. Layer-wise Attention Similarity Matrix Visualization
                # Ensure attn_map_one_step contains tensors for all model layers
                if len(attn_map_one_step) != num_model_layers:
                    print(
                        f"    Warning: Mismatch in expected layers ({num_model_layers}) and attentions received ({len(attn_map_one_step)}) for step {step_idx + 1}. Skipping similarity matrix for this step.")
                    continue

                layer_similarity_matrix = torch.zeros((num_model_layers, num_model_layers), device='cpu')
                for i in range(num_model_layers):
                    for j in range(num_model_layers):
                        # Get raw attention tensors for layer i and j at the current decode step
                        raw_attn_i = attn_map_one_step[i]
                        raw_attn_j = attn_map_one_step[j]

                        sim_score = cal_attn_map_similarity(raw_attn_i, raw_attn_j, top_k_tokens=SIMILARITY_TOP_K)
                        layer_similarity_matrix[i, j] = sim_score

                vis_layer_similarity_matrix(
                    sample_idx, step_idx, layer_similarity_matrix,
                    OUTPUT_VIS_DIR, PLOT_DPI
                )
        print("\nProcessing complete. Visualizations saved to:", OUTPUT_VIS_DIR)