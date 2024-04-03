from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn



def patch_attention(module):
    forward_orig = module.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        
        return forward_orig(*args, **kwargs)

    module.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        alignment = self.linear(x).squeeze(dim=-1)  # (n, 1) -> (n, )
        attention_weight = torch.softmax(alignment, dim=0)  # (n,)
        return attention_weight


class RDScanner(torch.nn.Module):
    """Semantic similarity calculating model with contrastive learning"""

    def __init__(
        self, input_dim: int = 1536, hidden_dim: int = 512, out_dim=128, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim

        self.attention_layer = AttentionLayer(hidden_dim)
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                dim_feedforward=hidden_dim,
                nhead=1,
            ),
        )
        self.context_layer = torch.nn.Linear(hidden_dim, out_dim)

    def represent(self, tensor: torch.Tensor) -> torch.Tensor:
        emb_patient = self.embedding(tensor)
        p_attention_weights = self.attention_layer(emb_patient)
        weighted_features = torch.einsum(
            "ij,ijk->ijk", p_attention_weights, emb_patient
        )

        p_context_vector = weighted_features.sum(axis=1)
        return self.context_layer(p_context_vector)

    def forward(
        self, patient_tensor: torch.Tensor, disease_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Return:
            representation (Tuple[torch.Tensor, torch.Tensor]):
                (batch, dim), (batch, dim)

        """
        if patient_tensor.ndim != 3 or disease_tensor.ndim != 3:
            raise ValueError(
                f"Passed patient_tensor dim ({patient_tensor.shape}), "
                f"and disease_tensor dim ({disease_tensor.shape})"
            )

        p_embedding = self.represent(patient_tensor)
        d_embedding = self.represent(disease_tensor)

        return p_embedding, d_embedding


class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        output_size,
        nhead,
        n_layers,
        batch_first=True,
    ):
        """
        Transformer-based model with InfoNCE loss for representation learning.

        Args:
            input_size (int): Size of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_size (int): Size of the output features.
            nhead (int): Number of heads in the multiheadattention models.
            n_layers (int): Number of sub-encoder-layers in the transformer encoder.
            temperature (float): Temperature parameter for scaling logits in InfoNCE loss.
            weight_decay (float): L2 regularization term.
            lr (float): Learning rate.

        Attributes:
            inp_layer (nn.Linear): Input layer.
            tf_encode (nn.TransformerEncoder): Transformer encoder.
            last_layer (nn.Linear): Final output layer.
            temperature (float): Temperature parameter for InfoNCE loss.
            weight_decay (float): L2 regularization term.
            lr (float): Learning rate.
            info_nce_loss (function): InfoNCE loss function.
            criteria (nn.CrossEntropyLoss): Cross-entropy loss function.
            _cached_vector (dict): Cached vectors for validation.
            train_loss (AverageMeter): Average training loss.
            validation_step_outputs (list): Validation step outputs.

        Example:
            >>> model = TransformerModel(256, 512, 128, 8, 3, 0.07, 1e-4, 0.0003)
        """
        super(Transformer, self).__init__()
        self.inp_layer = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=batch_first,
            dim_feedforward=hidden_dim,
        )
        self.batch_first = batch_first
        self.tf_encode = nn.TransformerEncoder(encoder_layer, n_layers)
        self.last_layer = nn.Sequential(nn.Linear(hidden_dim, output_size), nn.Tanh())

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # BxNxP
        x = self.inp_layer(x)
        if ~self.batch_first:
            x = x.permute(1, 0, 2)
        x = self.tf_encode(x)
        if ~self.batch_first:
            x = x.permute(1, 0, 2)
        x = self.last_layer(x)
        # BxD
        return x.mean(1)
    
    def get_att_weight(self, x):
        save_output = SaveOutput()
        for module in self.tf_encode.layers:
                patch_attention(module.self_attn)
                module.self_attn.register_forward_hook(save_output)
        with torch.no_grad():
            self(x)
        return save_output.outputs