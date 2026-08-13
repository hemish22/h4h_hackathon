"""Part 4.1-4.7 — gated fusion + multi-task model with aux heads and modality dropout."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn

from . import config as C


@dataclass
class ModelOutput:
    cls_logits: torch.Tensor            # (B,4)
    reg: torch.Tensor                   # (B,3) in [0,1]
    gate: torch.Tensor                  # (B,3) per-participant modality weights
    aux: Optional[List[torch.Tensor]]   # per-modality 4-class logits


class GatedFusion(nn.Module):
    """Projects each modality embedding to h, learns per-participant softmax gate
    over the 3 modalities, returns weighted concat + the gate (explainability)."""
    def __init__(self, dims=(128, 128, 64), h=128):
        super().__init__()
        self.h = h
        self.proj = nn.ModuleList([nn.Linear(d, h) for d in dims])
        self.gate = nn.Sequential(nn.Linear(h * 3, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, embs):
        z = [p(e) for p, e in zip(self.proj, embs)]
        g = torch.softmax(self.gate(torch.cat(z, 1)), 1)          # (B,3)
        fused = torch.cat([z[i] * g[:, i:i + 1] for i in range(3)], 1)
        return fused, g


class MultiTaskModel(nn.Module):
    """face + voice + tabular -> gated fusion -> trunk -> {cls head, 3 reg heads}
    plus per-modality aux heads (deep supervision, free ablation, concordance)."""
    def __init__(self, face_enc, voice_enc, tab_enc, h=128, p=0.3,
                 modality_dropout=0.15):
        super().__init__()
        self.face, self.voice, self.tab = face_enc, voice_enc, tab_enc
        dims = (face_enc.out_dim, voice_enc.out_dim, tab_enc.out_dim)
        self.fusion = GatedFusion(dims, h)
        trunk_in = h * 3
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(p),
        )
        self.cls_head = nn.Linear(128, C.N_CLASSES)
        self.reg_head = nn.Linear(128, 3)                        # 3 severity scores
        # aux heads read raw per-modality embeddings
        self.aux_heads = nn.ModuleList([nn.Linear(d, C.N_CLASSES) for d in dims])
        self.modality_dropout = modality_dropout

    def _encode(self, face, voice, tab):
        return [self.face(face), self.voice(voice), self.tab(tab)]

    def _apply_modality_dropout(self, embs):
        if not self.training or self.modality_dropout <= 0:
            return embs
        # never drop all three at once
        keep = [torch.rand(1).item() > self.modality_dropout for _ in embs]
        if not any(keep):
            keep[torch.randint(0, 3, (1,)).item()] = True
        return [e if k else torch.zeros_like(e) for e, k in zip(embs, keep)]

    def forward(self, face, voice, tab):
        embs = self._encode(face, voice, tab)
        aux = [h(e) for h, e in zip(self.aux_heads, embs)]
        embs = self._apply_modality_dropout(embs)
        fused, gate = self.fusion(embs)
        t = self.trunk(fused)
        reg = torch.sigmoid(self.reg_head(t))                    # [0,1]
        return ModelOutput(cls_logits=self.cls_head(t), reg=reg, gate=gate, aux=aux)
