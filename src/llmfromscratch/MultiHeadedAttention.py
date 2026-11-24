import torch
import torch.nn as nn

"""
#CUSTOM IMPLEMENTATION:-
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        #
        # Example: (1, 3, 4) -> (1, 3, 2, 2)
        #
        # BEFORE VIEW:
        #
        # TOKEN 0 = [1, 2, 3, 4]
        # TOKEN 1 = [5, 6, 7, 8]
        # TOKEN 2 = [9, 10, 11, 12]
        #
        # Each token has 4 features.
        #
        # AFTER VIEW:
        #
        # TOKEN 0:
        #   Head 0 = [1, 2]
        #   Head 1 = [3, 4]
        #
        # TOKEN 1:
        #   Head 0 = [5, 6]
        #   Head 1 = [7, 8]
        #
        # TOKEN 2:
        #   Head 0 = [9, 10]
        #   Head 1 = [11, 12]
        #
        # WHY?
        # Multi-head attention divides the embedding features into
        # multiple smaller groups called attention heads.
        #
        # Each head receives only a portion of the token's features.
        # These heads then independently calculate attention.
        #
        # VIEW changes the tensor's shape without changing its values.
        # It does not create or calculate new features.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # Example: (1, 3, 2, 2) -> (1, 2, 3, 2)
        #
        # BEFORE TRANSPOSE:
        #
        # TOKEN 0:
        #   Head 0 = [1, 2]
        #   Head 1 = [3, 4]
        #
        # TOKEN 1:
        #   Head 0 = [5, 6]
        #   Head 1 = [7, 8]
        #
        # TOKEN 2:
        #   Head 0 = [9, 10]
        #   Head 1 = [11, 12]
        #
        # AFTER TRANSPOSE:
        #
        # HEAD 0:
        #   Token 0 = [1, 2]
        #   Token 1 = [5, 6]
        #   Token 2 = [9, 10]
        #
        # HEAD 1:
        #   Token 0 = [3, 4]
        #   Token 1 = [7, 8]
        #   Token 2 = [11, 12]
        #
        # WHY?
        # We rearrange the tensor so that all token representations
        # belonging to the same attention head are grouped together.
        #
        # Each head now has a (num_tokens, head_dim) matrix.
        # This allows PyTorch to calculate attention independently
        # for every head using batched matrix multiplication.
        #
        # The values remain unchanged; only their dimension order changes.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, need_weights=True):
        super().__init__()
        self.context_length = context_length
        self.in_proj = nn.Linear(d_in, d_out, bias=qkv_bias) if d_in != d_out else nn.Identity()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )
        self.need_weights = need_weights
        self.proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        x = self.in_proj(x)                                    
        attn_mask = self.mask[:num_tokens, :num_tokens]
        attn_output, _ = self.multihead_attn(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )
        return self.proj(attn_output)

# #Test
if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )

    batch = torch.stack((inputs, inputs), dim=0)

    d_in , d_out = batch.shape[-1] , 2
    context_length = 128

    mha = MultiHeadAttention(
        d_in, d_out, 2, context_length, 0.0
    )

    context_vecs = mha(batch)

    print("context_vecs.shape:", context_vecs.shape)
    print("Context Vectors:-")
    print(context_vecs)

