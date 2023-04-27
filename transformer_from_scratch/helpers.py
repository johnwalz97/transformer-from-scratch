import re

import numpy as np


def load_dataset(file_path: str):
    """Load text dataset from file_path

    Args:
        file_path (str): path to file
    Returns:
        (str): text dataset
    """
    with open(file_path, "r") as f:
        return f.read()


def tokenize(text: str):
    """Tokenize text into list of words

    Args:
        text (str): text to tokenize
    Returns:
        (list): list of words
    """
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

def create_vocabulary(tokens: list):
    """Create vocabulary from list of tokens

    Args:
        tokens (list): list of tokens
    Returns:
        (dict): vocabulary
    """
    vocab = set(tokens)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for idx, token in enumerate(vocab)}
    return vocab, token_to_idx, idx_to_token


def text_to_indices(text: str, token_to_idx: dict):
    """Convert text to indices

    Args:
        text (str): text to convert
        token_to_idx (dict): token to index mapping
    Returns:
        (list): list of indices
    """
    tokens = tokenize(text)
    indices = [token_to_idx[token] for token in tokens]
    return np.array(indices)


def create_input_output_pairs(indices: np.ndarray, seq_len: int):
    """Create input-output pairs from indices

    Args:
        indices (np.ndarray): indices
        seq_len (int): sequence length
    Returns:
        (np.ndarray): input indices
        (np.ndarray): output indices
    """
    num_samples = len(indices) - seq_len
    input_data = np.zeros((num_samples, seq_len), dtype=int)
    output_data = np.zeros((num_samples, seq_len), dtype=int)
    
    for i in range(num_samples):
        input_data[i] = indices[i:i + seq_len]
        output_data[i] = indices[i + 1:i + seq_len + 1]

    return input_data, output_data


def one_hot_encode(indices: np.ndarray, vocab_size: int):
    """One hot encode indices

    Args:
        indices (np.ndarray): indices
        vocab_size (int): vocabulary size
    Returns:
        (np.ndarray): one hot encoded indices
    """
    one_hot = np.zeros((indices.shape[0], indices.shape[1], vocab_size))
    one_hot[
        np.arange(indices.shape[0])[:, None],
        np.arange(indices.shape[1]),
        indices,
    ] = 1

    return one_hot


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray):
    """Compute cross entropy loss

    Args:
        logits (np.ndarray): logits
        targets (np.ndarray): targets
    Returns:
        (float): cross entropy loss
    """
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    
    logits = logits[np.arange(batch_size)[:, None], np.arange(seq_len), targets]
    logits = np.clip(logits, 1e-12, None)  # Clip to avoid log(0)
    loss = -np.sum(np.log(logits)) / batch_size

    return loss


def softmax(x: np.ndarray):
    """Compute the softmax function for the input tensor.

    Args:
        x (np.ndarray): The input tensor.
    Returns:
        x (np.ndarray): The output tensor.
    """
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)
