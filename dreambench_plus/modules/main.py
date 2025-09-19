# ------------------------ knowledge_injection.py -----------------------
import torch, math, weakref
import torch.nn as nn
import torch
from enum import Enum
from typing import Optional, Union
import torch.nn as nn
from types import MethodType
class InjectionMethod(str, Enum):
    PROJECT_REPLACE     = "project_replace"
    RESIDUAL_PROJECTION = "residual_projection"
    ADDITIVE_UPDATE     = "additive_update"   # classic LoRA

class KnowledgeInjectionLayer(nn.Module):
    """
    LoRA‑style low‑rank injection that **never** registers the parent layer
    as a sub‑module, so no recursion is possible.
    """
    def __init__(
        self,
        base_layer: nn.Linear,                 # only Linear for brevity
        r: int = 8,
        injection_method: Union[str, InjectionMethod] = InjectionMethod.RESIDUAL_PROJECTION,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        init_scale: float = 1.0,
        use_gating: bool = False,
        decomposition_method: str = "qr",
        compute_svd_each_forward: bool = True,
        orthonormalize = False,
        device = None,
        train=False
    ):
        super().__init__()

        # --------‑  KEEP ONLY A FROZEN COPY OF THE WEIGHT  -----------
        self.register_buffer(
            "W_orig",
            base_layer.weight.detach().clone().to(device if device is not None else base_layer.weight.device),   # (out, in)
            persistent=False,                     # not saved in state_dict by default
        )
        if base_layer.bias is not None:
            self.register_buffer(
                "b_orig", base_layer.bias.detach().clone().to(device if device is not None else base_layer.bias.device), persistent=False
            )
        else:
            self.b_orig = None
        # --------------------------------------------------------------

        self.r   = r
        self.injection_method = InjectionMethod(injection_method)
        self.scaling = lora_alpha / r
        self.use_gating = use_gating
        self.compute_svd_each_forward = compute_svd_each_forward
        self.decomposition_method = decomposition_method

        # Optional dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.ortho = orthonormalize
        out_features, in_features = self.W_orig.shape
        # dtype = base_layer.weight.dtype        # fp16 under autocast / GradScaler
        dtype = torch.float32 if train else base_layer.weight.dtype         # fp32 for training
        dev   = base_layer.weight.device
        # learnable factors / projections
        self.P      = nn.Parameter(torch.zeros(out_features, r, device=dev, dtype=dtype))
        if self.use_gating!="0":
            self.R_new  = nn.Parameter(torch.zeros(r, in_features, device=dev, dtype=dtype))   
        if self.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            self.lora_A = nn.Parameter(torch.zeros(out_features, r, device=dev, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(r, in_features, device=dev, dtype=dtype))

        if use_gating:
            self.gate = nn.Parameter(torch.zeros(1, device=dev, dtype=dtype))   # start closed

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.P, mean=0.0, std=0.02)
        if self.injection_method != InjectionMethod.ADDITIVE_UPDATE:
            if self.use_gating!="0":
                nn.init.normal_(self.R_new, mean=0.0, std=0.02)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    # ---- small helper -------------------------------------------------
    def _orthogonalize(self, P: torch.Tensor) -> torch.Tensor:
        if self.decomposition_method == "qr":
            Q, _ = torch.linalg.qr(P.to(torch.float32))
            return Q[:, : self.r].to(P.dtype)
        elif self.decomposition_method == "svd":
            U, _, _ = torch.linalg.svd(P.to(torch.float32), full_matrices=False)
            return U[:, : self.r].to(P.dtype)
        else:
            raise ValueError(f"Unknown decomposition method {self.decomposition_method}")
    # -------------------------------------------------------------------

    # The layer now returns **only the update** for ADDITIVE_UPDATE
    # and the **full projected output** for the other two modes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dropped = self.lora_dropout(x)

        if self.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            update = (
                (x_dropped @ self.lora_B.T) @ self.lora_A.T
            ) * self.scaling
            if self.use_gating:
                update = update * torch.sigmoid(self.gate)
            return update

        # ----- methods that build W_modified ---------------------------------
        if self.ortho:
            # print(self.ortho)
            Q = self._orthogonalize(self.P) # if self.compute_svd_each_forward else self._orthogonalize(self.P.detach())
            P_proj = Q @ Q.T          # (out, out)
        else:
            Q = self.P
            P_proj = self.P @ (self.P).T          # (out, out)
        W_keep   = self.W_orig - P_proj @ self.W_orig
        # W_keep = self.W_orig - (P_proj.to(self.W_orig.dtype) @ self.W_orig) # silent the error expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
        

        if self.use_gating=="0":
            W_mod = W_keep
            # print("True")
        elif self.use_gating:
            W_inject = Q @ self.R_new
            W_inject = W_inject * torch.sigmoid(self.gate)
            W_mod = W_keep + W_inject  # (out, in)
        else:
            W_inject = Q @ self.R_new
            W_mod = W_keep + W_inject  # (out, in)
        out   = x_dropped @ W_mod.T
        if self.b_orig is not None:
            out = out + self.b_orig
        return out

def set_knowledge_injection_layer(self: nn.Linear, lora_layer: KnowledgeInjectionLayer):
    """
    Attach `inj_layer` to *this* Linear module and override its forward.
    No recursion because `inj_layer` **does not** register `self` as child.
    """
    # keep reference so it is saved with the parent
    self.lora_layer = lora_layer # keep the name → Diffusers happy

    if not hasattr(self, "_original_forward"):
        self._original_forward = self.forward          # save once

    def wrapped_forward(_self, x, *a, **kw):
        if lora_layer.injection_method == InjectionMethod.ADDITIVE_UPDATE:
            # Classic LoRA: original + low‑rank delta
            return _self._original_forward(x, *a, **kw) + lora_layer(x)
        else:
            # Project‑replace / residual‑projection replaces entire affine op
            return lora_layer(x)

    self.forward = MethodType(wrapped_forward, self)

# make it available globally
nn.Linear.set_knowledge_injection_layer = set_knowledge_injection_layer

