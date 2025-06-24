# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding implementation, as used in "Attention is All You Need".
    This injects information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CGEM(nn.Module):
    """
    The Complex Geometric Embedding Model (Order-Aware Version).
    This version uses a Transformer Encoder to process sequences of rich geometric feature vectors.
    """

    # --- UPDATED: Input dimension is now 8 to handle the new features ---
    def __init__(self, input_dim=8, model_dim=128, output_m=16, nhead=8, num_encoder_layers=4, dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)

        # --- NEW: Add the PositionalEncoding layer ---
        self.pos_encoder = PositionalEncoding(d_model=model_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: Input shape is (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.output_head = nn.Linear(model_dim, 2 * output_m)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (B, seq_len, input_dim) e.g., (B, 128, 8)

        # 1. Embed input and scale
        x = self.input_embedding(src) * math.sqrt(self.model_dim)

        # 2. Add positional encoding
        # Note: We need to transpose for the standard PositionalEncoding layer
        x = x.transpose(0, 1)  # (seq_len, B, model_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, seq_len, model_dim)

        # 3. Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # 4. Aggregate features (pooling)
        x = x.mean(dim=1)  # Global average pooling -> (B, model_dim)

        # 5. Map to complex embedding space
        x = self.output_head(x)  # (B, 2 * m)

        # Reshape to (B, m, 2) and convert to complex type
        x = x.view(x.size(0), -1, 2)
        complex_out = torch.view_as_complex(x)

        # Normalize the complex vectors to lie on the unit hypersphere
        return F.normalize(complex_out, p=2, dim=1)


class ComplexLoss(nn.Module):
    """Computes the combined loss for rotation and similarity."""

    def __init__(self, lambda_sim=1.0):
        super().__init__()
        self.lambda_sim = lambda_sim

    def forward(self, u1: torch.Tensor, u2: torch.Tensor, rot_target: torch.Tensor,
                sim_target: torch.Tensor) -> torch.Tensor:
        # u1, u2 shapes: (B, m) of complex type
        pred_dot = torch.sum(u1 * u2.conj(), dim=1)

        # Rotation Loss
        pred_angle = torch.angle(pred_dot)
        loss_rot = 1 - torch.pow(torch.cos(pred_angle - rot_target), 2)

        # Similarity Loss
        pred_mag = torch.abs(pred_dot)
        loss_sim = F.mse_loss(pred_mag, sim_target)

        total_loss = torch.mean(loss_rot) + self.lambda_sim * loss_sim
        return total_loss


if __name__ == '__main__':
    """
    Simple test block for the model and loss function.
    """
    print("--- Running model.py self-test (Order-Aware Version) ---")

    # Test parameters
    batch_size = 4
    n_resample = 128
    # --- UPDATED: input_dim is now 8 ---
    input_dim = 8
    output_m = 16

    # 1. Model forward pass test
    model = CGEM(input_dim=input_dim)
    dummy_input = torch.randn(batch_size, n_resample, input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, output_m)
    print(f"[PASS] Model output shape is correct: {output.shape}")
    assert output.dtype == torch.complex64
    print(f"[PASS] Model output dtype is correct: {output.dtype}")

    # 2. Loss function test
    loss_fn = ComplexLoss(lambda_sim=1.0)
    dummy_u1 = model(torch.randn(batch_size, n_resample, input_dim))
    dummy_u2 = model(torch.randn(batch_size, n_resample, input_dim))
    dummy_rot = torch.rand(batch_size) * 2 * math.pi
    dummy_sim = torch.rand(batch_size)

    loss = loss_fn(dummy_u1, dummy_u2, dummy_rot, dummy_sim)

    assert loss.ndim == 0  # Loss should be a scalar
    print("[PASS] Loss function produced a scalar output.")
    assert loss.item() >= 0
    print(f"[PASS] Loss function produced a non-negative value: {loss.item():.4f}")

    print("--- model.py self-test complete ---")
