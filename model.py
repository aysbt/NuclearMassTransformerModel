import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, categorical_feature_sizes: list):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(feature_size, d_model) for feature_size in categorical_feature_sizes
        ])

    def forward(self, x):
        embedded_features = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        return torch.stack(embedded_features, dim=1)  # (batch, num_cat_features, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.layer_norm = nn.LayerNorm(d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        return self.fc2(self.dropout(x))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.layer_norm(out)
        return self.w_o(out), attn

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        self.last_attention_weights = []  # initialize storage for attention
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        self.last_attention_weights = []          # reset for this forward pass
        for layer in self.layers:
            x, attn = layer(x, mask) # attn shape: (batch, heads, seq_len, seq_len)
            self.last_attention_weights.append(attn.detach())
        out = self.norm(x)
        return out, self.last_attention_weights
    
    def get_attention_weights(self):
        return self.last_attention_weights 


class TransformerMassExcessPredictor(nn.Module):
    def __init__(self, num_cont_features, categorical_feature_sizes, d_model=128, num_heads=8, d_ff=512, num_layers=4, dropout=0.10):
        super().__init__()
        self.has_categorical = len(categorical_feature_sizes) > 0
        self.categorical_embedding = InputEmbeddings(d_model, categorical_feature_sizes) if self.has_categorical else None

        # MLP for continuous features
        self.continuous_embedding = nn.Sequential(
            nn.Linear(num_cont_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        max_seq_len = len(categorical_feature_sizes)
        self.positional_encoding = PositionalEncoding(d_model, seq_len=max_seq_len, dropout=dropout)

        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.ffn = FeedForwardBlock(d_model * 2, d_ff, dropout)
        self.baseline_layer = nn.Linear(num_cont_features, 1)

        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, categorical_inputs, continuous_inputs):
        if self.has_categorical:
            cat_tokens = self.categorical_embedding(categorical_inputs)
            cat_tokens = self.positional_encoding(cat_tokens)
            encoded_cat, attention_weights = self.encoder(cat_tokens)
            pooled_cat = encoded_cat.mean(dim=1)
        else:
            pooled_cat = torch.zeros(continuous_inputs.size(0), self.continuous_embedding[-1].out_features).to(continuous_inputs.device)
            attention_weights = None

        cont_vector = self.continuous_embedding(continuous_inputs)  # (batch, d_model)
        

        # Concatenate both
        fused = torch.cat([pooled_cat, cont_vector], dim=1)  # (batch, d_model * 2)
        fused = self.ffn(fused)

        #predicted_mass_excess = self.regressor(fused).squeeze(1)
        # Add Residual Connection to Prediction
        #predicted_mass_excess = self.regressor(fused).squeeze(1) + cont_vector[:, 0]
        baseline = self.baseline_layer(continuous_inputs).squeeze(1)
        predicted_mass_excess = self.regressor(fused).squeeze(1) + baseline
        return predicted_mass_excess, attention_weights