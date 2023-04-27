import numpy as np

from .helpers import softmax


class NumpyGPT:
    """A simple implementation of a transformer model using NumPy."""

    def __init__(self, vocab_size, d_model, nhead, num_layers):
        """Initialize the NumpyGPT model with the provided parameters.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the model.
            nhead (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.embedding = np.random.randn(vocab_size, d_model) * np.sqrt(1 / vocab_size)
        self.transformer_layers = [
            TransformerLayer(d_model, nhead) for _ in range(num_layers)
        ]
        self.fc_out = np.random.randn(d_model, vocab_size) * np.sqrt(1 / d_model)

    def forward(self, x):
        """Perform a forward pass through the model.

        Args:
            x (np.ndarray): The input sequence of shape (batch_size, seq_len).
        Returns:
            logits (np.ndarray): The logits of shape (batch_size, seq_len, vocab_size).
        """
        # Input embedding
        x = np.matmul(x, self.embedding)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer.forward(x)

        # Final linear layer
        logits = np.matmul(x, self.fc_out)

        return logits

    def backward(self, logits, targets):
        """Perform a backward pass through the model.

        Args:
            logits (np.ndarray): The logits from the forward pass of shape (batch_size, seq_len, vocab_size).
            targets (np.ndarray): The target sequence of shape (batch_size, seq_len, vocab_size).
        Returns:
            d_embedding (np.ndarray): Gradient of the embedding layer.
            d_transformer_layers (list): List of gradients for each transformer layer.
            d_fc_out (np.ndarray): Gradient of the final linear layer.
        """
        batch_size, seq_len, _ = logits.shape

        # compute softmax gradients
        d_logits = softmax(logits)
        # subtract 1 from the target positions
        d_logits[np.arange(batch_size)[:, None], np.arange(seq_len), targets] -= 1
        # divide by batch size
        d_logits /= batch_size

        # compute gradients for the final linear layer
        d_fc_out = np.matmul(
            self.transformer_layers[-1].x.reshape(-1, self.d_model).T,
            d_logits.reshape(-1, self.vocab_size),
        )
        d_transformer_layers = []

        # compute gradients for each transformer layer
        for layer in reversed(self.transformer_layers):
            d_x = layer.backward(d_logits)
            d_transformer_layers.append(layer.get_gradients())

        # compute gradients for the embedding layer
        d_embedding = np.matmul(d_x, self.embedding.T)

        return d_embedding, d_transformer_layers, d_fc_out


class TransformerLayer:
    """A single transformer layer consisting of self-attention and a feed-forward network."""

    def __init__(self, d_model, nhead):
        """Initialize the TransformerLayer with the provided parameters.

        Args:
            d_model (int): The dimensionality of the model.
            nhead (int): The number of attention heads.
        """
        self.attention = MultiHeadAttention(d_model, nhead)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        """Perform a forward pass through the layer.

        Args:
            x (np.ndarray): The input sequence of shape (batch_size, seq_len, d_model).
        Returns:
            x (np.ndarray): The output sequence of shape (batch_size, seq_len, d_model).
        """
        # Self-attention and add & norm
        att_output = self.attention.forward(x, x, x)
        self.x = self.norm1.forward(x + att_output)

        # Feed-forward and add & norm
        ff_output = self.feed_forward.forward(x)
        self.x = self.norm2.forward(x + ff_output)

        return self.x

    def backward(self, d_x):
        """Perform a backward pass through the layer.

        Args:
            d_x (np.ndarray): Gradient from the next layer of shape (batch_size, seq_len, d_model).
        Returns:
            d_x (np.ndarray): Gradient of the input tensor of shape (batch_size, seq_len, d_model).
        """
        d_ff_input = self.norm2.backward(d_x)
        d_x = self.feed_forward.backward(d_ff_input)

        d_att_input = self.norm1.backward(d_x)
        d_x = self.attention.backward(d_att_input)

        return d_x

    def get_gradients(self):
        """Get gradients of the TransformerLayer components.

        Returns:
            gradients (tuple): Gradients of attention, norm1, feed_forward, and norm2.
        """
        return (
            self.attention.get_gradients(),
            self.norm1.get_gradients(),
            self.feed_forward.get_gradients(),
            self.norm2.get_gradients(),
        )


class MultiHeadAttention:
    """Multi-head attention mechanism for the transformer model."""

    def __init__(self, d_model, nhead):
        """Initialize the MultiHeadAttention with the provided parameters.

        Args:
            d_model (int): The dimensionality of the model.
            nhead (int): The number of attention heads.
        """
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01

    def _split_heads(self, x, batch_size):
        """Split the last dimension of the input tensor into (nhead, d_k).

        Args:
            x (np.ndarray): The input tensor of shape (batch_size, seq_len, d_model).
            batch_size (int): The batch size.
        Returns:
            x (np.ndarray): The output tensor of shape (batch_size, nhead, seq_len, d_k).
        """
        x = np.reshape(x, (batch_size, -1, self.nhead, self.d_k))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, query, key, value):
        """Perform a forward pass through the multi-head attention mechanism.

        Args:
            query (np.ndarray): The query tensor of shape (batch_size, seq_len, d_model).
            key (np.ndarray): The key tensor of shape (batch_size, seq_len, d_model).
            value (np.ndarray): The value tensor of shape (batch_size, seq_len, d_model).
        Returns:
            context (np.ndarray): The context tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = query.shape[0]

        q = np.matmul(query, self.W_q)
        k = np.matmul(key, self.W_k)
        v = np.matmul(value, self.W_v)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Compute scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn_weights = softmax(scores)
        context = np.matmul(attn_weights, v)

        context = np.concatenate(np.split(context, self.nhead, axis=1), axis=-1)
        context = np.squeeze(context, axis=1)

        return context

    def backward(self, d_context):
        """Perform a backward pass through the multi-head attention mechanism.

        Args:
            d_context (np.ndarray): Gradient from the next layer of shape (batch_size, seq_len, d_model).
        Returns:
            d_query (np.ndarray): Gradient of the query tensor of shape (batch_size, seq_len, d_model).
        """
        d_v = np.matmul(self.attn_weights.T, d_context)
        d_attn_weights = np.matmul(d_context, self.v.T)

        d_scores = d_attn_weights * (1 - np.eye(self.scores.shape[1]))[None, :, :]
        d_q = np.matmul(d_scores, self.k) / np.sqrt(self.d_k)

        d_k = np.matmul(d_scores.transpose(0, 1, 3, 2), self.q) / np.sqrt(self.d_k)
        d_query = np.matmul(d_q + d_k, self.W_q.T)

        return d_query

    def get_gradients(self):
        """Get gradients of the MultiHeadAttention components.

        Returns:
            gradients (tuple): Gradients of W_q, W_k, and W_v.
        """
        return (
            np.matmul(self.query.T, self.q),
            np.matmul(self.key.T, self.k),
            np.matmul(self.value.T, self.v),
        )


class LayerNorm:
    """Layer normalization layer for the transformer model."""

    def __init__(self, hidden_dim, epsilon=1e-5):
        """Initialize the LayerNorm with the provided parameters.

        Args:
            hidden_dim (int): The dimensionality of the hidden state.
            epsilon (float): A small value used for numerical stability.
        """
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon

        # Initialize learnable parameters (gamma and beta) as arrays of ones and zeros, respectively
        self.gamma = np.ones(hidden_dim)
        self.beta = np.zeros(hidden_dim)

        # Initialize the gradients for the parameters
        self.d_gamma = np.zeros(hidden_dim)
        self.d_beta = np.zeros(hidden_dim)

    def forward(self, x):
        """Perform a forward pass through the layer normalization layer.

        Args:
            x (np.ndarray): The input tensor of shape (batch_size, seq_len, hidden_dim).
        Returns:
            out (np.ndarray): The output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        # Compute the mean and standard deviation
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)

        # Normalize the input
        self.normalized_x = (x - mean) / (std + self.epsilon)

        # Apply the learnable parameters (scale and shift)
        self.out = self.gamma * self.normalized_x + self.beta

        return self.out

    def backward(self, d_x):
        """Perform a backward pass through the layer normalization layer.

        Args:
            d_x (np.ndarray): Gradient from the next layer of shape (batch_size, seq_len, hidden_dim).
        Returns:
            dx (np.ndarray): Gradient of the input tensor of shape (batch_size, seq_len, hidden_dim).
        """
        *_, D = d_x.shape

        # Compute the gradients for gamma and beta
        self.d_gamma = np.sum(d_x * self.normalized_x, axis=(0, 1))
        self.d_beta = np.sum(d_x, axis=(0, 1))

        # Compute the gradient for the input
        dx_norm = d_x * self.gamma
        dstd = np.sum(dx_norm * (self.normalized_x), axis=-1, keepdims=True) * -1 / (self.epsilon + np.square(np.std(self.normalized_x, axis=-1, keepdims=True)))
        dmean = np.sum(dx_norm * -1 / (self.epsilon + np.std(self.normalized_x, axis=-1, keepdims=True)), axis=-1, keepdims=True) + dstd * np.mean(-2 * self.normalized_x, axis=-1, keepdims=True)
        d_x = dx_norm / (self.epsilon + np.std(self.normalized_x, axis=-1, keepdims=True)) + dstd * 2 * self.normalized_x / D + dmean / D

        return d_x

    def get_gradients(self):
        """Get gradients of the LayerNorm components.

        Returns:
            gradients (tuple): Gradients of gamma and beta.
        """
        return self.d_gamma, self.d_beta


class FeedForward:
    """A simple feed-forward network for the transformer model."""

    def __init__(self, d_model, d_ff=2048):
        """Initialize the FeedForward network with the provided parameters.

        Args:
            d_model (int): The dimensionality of the model.
            d_ff (int): The dimensionality of the feed-forward network.
                        (default: 2048)
        """
        self.fc1 = np.random.randn(d_model, d_ff) * np.sqrt(1 / d_model)
        self.fc2 = np.random.randn(d_ff, d_model) * np.sqrt(1 / d_ff)

    def forward(self, x):
        """Perform a forward pass through the feed-forward network.

        Args:
            x (np.ndarray): The input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            x (np.ndarray): The output tensor of shape (batch_size, seq_len, d_model).
        """
        # First fully connected layer with ReLU activation
        x = np.matmul(x, self.fc1)
        x = np.maximum(0, x)  # ReLU

        # Second fully connected layer
        x = np.matmul(x, self.fc2)

        return x

    def backward(self, d_x):
        """Perform a backward pass through the feed-forward network.

        Args:
            d_x (np.ndarray): Gradient from the next layer of shape (batch_size, seq_len, d_model).
        Returns:
            d_input (np.ndarray): Gradient of the input tensor of shape (batch_size, seq_len, d_model).
        """
        self.d_fc2 = np.matmul(self.relu_output.T, d_x)
        d_relu = np.matmul(d_x, self.fc2.T)

        self.d_fc1 = np.matmul(self.x.T, d_relu * (self.relu_output > 0))
        d_input = np.matmul(d_relu * (self.relu_output > 0), self.fc1.T)

        return d_input

    def get_gradients(self):
        """Get gradients of the FeedForward components.

        Returns:
            gradients (tuple): Gradients of fc1 and fc2.
        """
        return self.d_fc1, self.d_fc2
