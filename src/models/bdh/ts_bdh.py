import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BDH_ROOT = Path(__file__).resolve().parent / "BDH_GPU"
sys.path.insert(0, str(BDH_ROOT))

from model import BDH_GPU


class TSBDH(nn.Module):
    """
    Time-series wrapper around the original BDH_GPU implementation.

    The original BDH source is not modified.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        seq_len,
        pred_len,
        D=128,
        H=4,
        N=4096,
        L=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Input: numerical time-series -> BDH latent space
        self.input_projection = nn.Linear(input_dim, D)

        # Original BDH architecture
        self.bdh = BDH_GPU(
            vocab_size=1,
            D=D,
            H=H,
            N=N,
            L=L,
            dropout=dropout,
        )

        # We don't use BDH's token embedding/readout.
        # The internal BDH parameters remain unchanged.

        # Forecasting head
        self.forecast_head = nn.Linear(
    D,
    pred_len * output_dim,
)

    def forward(self, x):
        """
        x: [B, T, input_dim]

        returns:
            [B, pred_len, output_dim]
        """

        B, T, _ = x.shape

        # Numerical input -> BDH latent
        v_ast = self.bdh.ln(
            self.input_projection(x)
        )

        # Original BDH internal computation
        for _ in range(self.bdh.L):

            dec_x = (
                self.bdh.decoder_x
                * self.bdh.mask_decoder_x
            )

            x_lin = torch.einsum(
                "btd,hdn->bhtn",
                v_ast,
                dec_x,
            )

            x_act = F.relu(x_lin)

            # Original BDH attention
            a_ast = self.bdh.attn(
                Q=x_act,
                K=x_act,
                V=v_ast.unsqueeze(1),
            )

            dec_y = (
                self.bdh.decoder_y
                * self.bdh.mask_decoder_y
            )

            y_lin = torch.einsum(
                "btd,hdn->bhtn",
                self.bdh.ln(a_ast).squeeze(1),
                dec_y,
            )

            y = F.relu(y_lin) * x_act

            y = y.transpose(1, 2).reshape(
                B, 1, T, self.bdh.N
            )

            y = self.bdh.drop(y)

            enc = (
                self.bdh.encoder
                * self.bdh.mask_encoder
            )

            v_ast = v_ast.unsqueeze(1)

            v_ast = v_ast + self.bdh.ln(
                y @ enc
            )

            v_ast = self.bdh.ln(v_ast)

            v_ast = v_ast.squeeze(1)

        # Use final latent state.
        context = v_ast[:, -1, :]

        # Predict the complete horizon.
        forecast = self.forecast_head(context)

        forecast = forecast.view(
            B,
            self.pred_len,
            self.output_dim,
        )

        return forecast