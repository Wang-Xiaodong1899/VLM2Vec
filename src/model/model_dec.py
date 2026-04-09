from __future__ import annotations

from typing import Optional

from typing import Dict
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from src.model.vlm_backbone.qwen2_vl.modeling_qwen2_vl import (
    Qwen2RMSNorm,
    Qwen2VLDecoderLayer,
    Qwen2VLRotaryEmbedding,
)

from .model import MMEBModel
from src.utils.basic_utils import print_rank, print_master


class PostQwen2VLDecoderStack(nn.Module):
    def __init__(self, config, num_layers: int):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        self.config = config
        self.layers = nn.ModuleList([Qwen2VLDecoderLayer(config, i) for i in range(num_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if cache_position is None:
            cache_position = torch.arange(seq_len, device=hidden_states.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        return self.norm(hidden_states)

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0).any().item():
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else sequence_length + 1

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                sliding_attend_mask = torch.arange(target_length, device=device) <= (
                    cache_position.reshape(-1, 1) - config.sliding_window
                )
                diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class MMEBModelPostQwen2VLLayers(MMEBModel):
    def __init__(
        self,
        encoder,
        pooling: str = "last",
        normalize: bool = False,
        temperature: float = 0.02,
        post_qwen2vl_layers: int = 8, # 1, 4, 8
    ):
        super().__init__(encoder=encoder, pooling=pooling, normalize=normalize, temperature=temperature)
        self.post_decoder = PostQwen2VLDecoderStack(self.config, num_layers=post_qwen2vl_layers)
        first_param = next(self.encoder.parameters(), None)
        if first_param is not None:
            self.post_decoder.to(dtype=first_param.dtype)

    def encode_input(self, input, is_target: bool = False):
        from src.model.processor import COLPALI, GME, INTERNVIDEO2, LLAVA_NEXT, LamRA, LamRA_QWEN2_5

        model_backbone = getattr(self, "model_backbone", None)
        if model_backbone in [INTERNVIDEO2, GME, LamRA, LamRA_QWEN2_5, COLPALI]:
            return super().encode_input(input)

        if model_backbone == LLAVA_NEXT:
            input["pixel_values"] = input["pixel_values"].squeeze(dim=1)
            input["image_sizes"] = input["image_sizes"].squeeze(dim=1)

        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        encoder_trainable = any(p.requires_grad for p in self.encoder.parameters())
        if encoder_trainable:
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        else:
            with torch.no_grad():
                # print("encoder is not trainable, use no_grad")
                hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        if is_target:
            # no pass post decoder
            # print(f"tgt hidden_states.shape: {hidden_states.shape}")
            pooled_output = self._pooling(hidden_states, input["attention_mask"])
            return pooled_output
        hidden_states = self.post_decoder(
            hidden_states,
            attention_mask=input.get("attention_mask"),
            position_ids=input.get("position_ids"),
            cache_position=input.get("cache_position"),
        )
        # print(f"post decoder hidden_states.shape: {hidden_states.shape}")
        # print(f"qry involve hidden_states.shape: {hidden_states.shape}")
        pooled_output = self._pooling(hidden_states, input["attention_mask"])
        return pooled_output

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None,
        ans: Dict[str, Tensor] = None,
        qp_loss_weight: float = 1.0,
        qa_loss_weight: float = 0.0,
        pa_loss_weight: float = 0.0,
        *args, **kwargs):
        qry_reps = self.encode_input(qry, is_target=False) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt, is_target=False) if tgt else None # (bsz_per_device, dim)
        ans_reps = self.encode_input(ans, is_target=False) if ans else None # (bsz_per_device, dim)
        if qry_reps is None and tgt_reps is None and ans_reps is None:
            return {"qry_reps": None, "tgt_reps": None}
        if qry_reps is None and tgt_reps is None:
            return {"qry_reps": None, "tgt_reps": ans_reps}
        if qry_reps is None or (tgt_reps is None and ans_reps is None):
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}
        if tgt_reps is None and ans_reps is not None:
            return {"qry_reps": qry_reps, "tgt_reps": ans_reps}
        
        def _contrastive_loss(all_left: Tensor, all_right: Tensor) -> Tensor:
            scores = self.compute_similarity(all_left, all_right)
            scores = scores.view(all_left.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (all_left.size(0) // all_right.size(0))
            return self.cross_entropy(scores / self.temperature, target)

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            all_ans_reps = self._dist_gather_tensor(ans_reps) if ans_reps is not None else None
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps
            all_ans_reps = ans_reps

        # scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        # scores = scores.view(all_qry_reps.size(0), -1)
        # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        # target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        # loss = self.cross_entropy(scores / self.temperature, target)
        total_loss = torch.zeros((), device=all_qry_reps.device)
        if all_tgt_reps is not None and qp_loss_weight != 0:
            # total_loss = total_loss + float(qp_loss_weight) * _contrastive_loss(all_qry_reps, all_tgt_reps)
            qp_loss = float(qp_loss_weight) * _contrastive_loss(all_qry_reps, all_tgt_reps)
            total_loss = total_loss + qp_loss
        else:
            qp_loss = None
        if all_ans_reps is not None and qa_loss_weight != 0:
            # total_loss = total_loss + float(qa_loss_weight) * _contrastive_loss(all_qry_reps, all_ans_reps)
            qa_loss = float(qa_loss_weight) * _contrastive_loss(all_qry_reps, all_ans_reps)
            total_loss = total_loss + qa_loss
        else:
            qa_loss = None
        if all_ans_reps is not None and pa_loss_weight != 0:
            # total_loss = total_loss + float(pa_loss_weight) * _contrastive_loss(all_tgt_reps, all_ans_reps)
            pa_loss = float(pa_loss_weight) * _contrastive_loss(all_tgt_reps, all_ans_reps)
            total_loss = total_loss + pa_loss
        else:
            pa_loss = None
        print_rank(f"total_loss: {total_loss}, qp_loss: {qp_loss}, qa_loss: {qa_loss}, pa_loss: {pa_loss}")
        if self.is_ddp:
            # loss = loss * self.world_size
            total_loss = total_loss * self.world_size

        return total_loss
