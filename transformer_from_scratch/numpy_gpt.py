import numpy as np


class NumpyGPT:
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.transformer_layers = [
            TransformerLayer(d_model, nhead) for _ in range(num_layers)
        ]
        self.fc_out = np.random.randn(d_model, vocab_size) * 0.01

    def forward(self, x):
        # Input embedding
        x = np.matmul(x, self.embedding)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer.forward(x)

        # Final linear layer
        logits = np.matmul(x, self.fc_out)

        return logits


class TransformerLayer:
    def __init__(self, d_model, nhead):
        self.attention = MultiHeadAttention(d_model, nhead)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x):
        # Self-attention and add & norm
        att_output = self.attention.forward(x, x, x)
        x = self.norm1.forward(x + att_output)

        # Feed-forward and add & norm
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)

        return x


class MultiHeadAttention:
    def __init__(self, d_model, nhead):
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_q = np.random.randn(nhead, d_model, self.d_k) * 0.01
        self.W_k = np.random.randn(nhead, d_model, self.d_k) * 0.01
        self.W_v = np.random.randn(nhead, d_model, self.d_k) * 0.01

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        q = np.matmul(query, self.W_q)
        k = np.matmul(key, self.W_k)
        v = np.matmul(value, self.W_v)

        # Compute scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        attn_weights = softmax(scores)
        context = np.matmul(attn_weights, v)

        return context


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward:
    def __init__(self, d_model, d_ff=2048):
        self.fc1 = np.random.randn(d_model, d_ff) * 0.01
        self.fc2 = np.random.randn(d_ff, d_model) * 0.01

    def forward(self, x):
        # First fully connected layer with ReLU activation
        x = np.matmul(x, self.fc1)
        x = np.maximum(0, x)  # ReLU

        # Second fully connected layer
        x = np.matmul(x, self.fc2)

        return x


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
