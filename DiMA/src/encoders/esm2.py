import os
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from typing import Optional, Dict, List
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.encoders.transformer_decoder import TransformerDecoder
from src.encoders.base import Encoder


class ESM2EncoderModel(Encoder):
    def __init__(
            self,
            config: DictConfig,
            main_config: DictConfig = None,
            device: torch.device = None,
            add_enc_normalizer: bool = True,
    ):  
        super().__init__(
            config=config,
            device=device,
            decoder_type=main_config.decoder.decoder_type,
            add_enc_normalizer=add_enc_normalizer,
        )
        self.main_config = main_config
        self._use_transformer_decoder = False
        self._special_token_ids = set()

        esm_model = EsmForMaskedLM.from_pretrained(
            config.encoder_model_name, 
            add_cross_attention=False, 
            is_decoder=False
        )

        self.tokenizer = EsmTokenizer.from_pretrained(config.encoder_model_name)
        self._special_token_ids = set(self.tokenizer.all_special_ids)
        self.encoder = esm_model.esm.to(device)
        self.lm_head_decoder = esm_model.lm_head.to(device)
        
        if self.decoder_type == "transformer":
            decoder_path = self.main_config.decoder.decoder_path
            if decoder_path is not None and os.path.exists(decoder_path):
                self.sequence_decoder = TransformerDecoder(
                    config=self.main_config
                )
                self.sequence_decoder.load_state_dict(torch.load(decoder_path)["decoder"])
                self._use_transformer_decoder = True
            else:
                # Fall back to ESM lm_head if a trained transformer decoder is unavailable.
                # Using a randomly initialized decoder can yield mostly empty invalid outputs.
                print("Decoder checkpoint was not found; falling back to ESM lm_head for decoding")
                self.sequence_decoder = self.lm_head_decoder
        else:
            self.sequence_decoder = self.lm_head_decoder
        self.sequence_decoder = self.sequence_decoder.to(device)

    def _decode_logits_to_sequences(self, logits: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> List[str]:
        # Avoid decoding to tokenizer special tokens (<pad>/<cls>/<eos>/...),
        # which become empty strings when skip_special_tokens=True.
        logits_orig = logits
        logits = logits.clone()
        
        # Debug: check what tokens are winning before masking
        top_tokens_pre = logits[0, :5].argsort(descending=True).tolist()
        
        for token_id in self._special_token_ids:
            logits[..., token_id] = torch.finfo(logits.dtype).min
        
        # Debug: check what tokens are winning after masking
        top_tokens_post = logits[0, :5].argsort(descending=True).tolist()
        max_val = logits.max().item()
        print(f"[DEBUG] Logits max before special-mask: {logits_orig.max().item():.4f}, after: {max_val:.4f}")
        print(f"[DEBUG] Top 5 tokens before mask: {top_tokens_pre}, after: {top_tokens_post}")

        token_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        if attention_mask is not None:
            for i, t in enumerate(token_ids):
                seq_len = int(attention_mask[i].sum().item())
                token_ids[i] = t[:seq_len]

        decoded_raw = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [''.join(t.split()) for t in decoded_raw]
        
    def batch_encode(self, batch: Dict, max_sequence_len: int):
        max_len_with_special_tokens = max_sequence_len + 2

        sequences = batch["sequence"]
        tokenized_batch = self.tokenizer(
            sequences, 
            return_attention_mask=True, 
            return_tensors="pt", 
            truncation=True,                       
            padding=True, 
            max_length=max_len_with_special_tokens,
            return_special_tokens_mask=True,
        )
        tokenized_batch = tokenized_batch.to(self.device)

        encodings = self.encoder(
            input_ids=tokenized_batch["input_ids"], 
            attention_mask=tokenized_batch["attention_mask"]
        ).last_hidden_state
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.normalize(encodings)

        return encodings, tokenized_batch["attention_mask"], tokenized_batch["input_ids"]

    def batch_decode(self, encodings, attention_mask=None):
        encodings = self.enc_normalizer.denormalize(encodings)
        if self._use_transformer_decoder:
            logits = self.sequence_decoder(x=encodings, mask=attention_mask)
        else:
            logits = self.sequence_decoder(encodings)

        decoded_sequences = self._decode_logits_to_sequences(logits, attention_mask=attention_mask)

        # Safety fallback: if transformer decoding collapses to all-special tokens,
        # decode with lm_head to avoid silently producing all-empty outputs.
        if self._use_transformer_decoder and decoded_sequences and all(len(seq) == 0 for seq in decoded_sequences):
            print("[WARN] Transformer decoder produced only empty sequences; falling back to ESM lm_head for this batch")
            logits = self.lm_head_decoder(encodings)
            decoded_sequences = self._decode_logits_to_sequences(logits, attention_mask=attention_mask)

        return decoded_sequences  
        

    def batch_get_logits(self, encodings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enc_normalizer is not None:
            encodings = self.enc_normalizer.denormalize(encodings)
        if self._use_transformer_decoder:
            logits = self.sequence_decoder(x=encodings, mask=attention_mask)
        else:
            logits = self.sequence_decoder(encodings)
        return logits

    def restore_decoder(self, decoder_path: str):
        if os.path.exists(decoder_path):
            self.sequence_decoder.load_state_dict(torch.load(decoder_path)["decoder"])
        else:
            print(f"Warning: Decoder checkpoint path provided, but no decoder is present in the model.")