import os
import re
import sys
import glob
import json


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class ExtractOps:
    def __init__(self, config_file):
        self.load_config(config_file) 
    
    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f) 
        self.config = config
        self.dtype = config['torch_dtype']
        self.hidden_size = self.config['hidden_size']
        if self.config.get('head_dim', None):
            self.head_dim = self.config['head_dim']
        else:
            self.head_dim = divide_and_check_no_remainder(
                self.hidden_size, self.config['num_attention_heads'])
        self.moe_intermediate_size = self.config.get('moe_intermediate_size', None)
    
    def extract_attention_gemm(self, m=1, tp_size=1):
        q_n = self.head_dim * self.config['num_attention_heads']
        q_n = divide_and_check_no_remainder(q_n, tp_size)
        kv_n = self.head_dim * self.config['num_key_value_heads']
        kv_n = divide_and_check_no_remainder(kv_n, tp_size)
        qkv_k = self.hidden_size
        results = [
            f"m={m}, n={q_n}, k={qkv_k}, dtype={self.dtype}, flag=q_proj", # q
            f"m={m}, n={kv_n}, k={qkv_k}, dtype={self.dtype}, flag=kv_proj", # kv
            f"m={m}, n={qkv_k}, k={q_n}, dtype={self.dtype}, flag=o_proj", # o
        ]
        return sorted(list(set(results)))

    def extract_ffn_gemm(self, m=1, tp_size=1):
        if self.moe_intermediate_size:
            intermediate_size = self.moe_intermediate_size
        else:
            intermediate_size = self.config['intermediate_size']
        gate_up_n = divide_and_check_no_remainder(intermediate_size, tp_size)
        gate_up_k = self.hidden_size
        down_n = gate_up_k
        down_k = gate_up_n
        results = [
            f"m={m}, n={gate_up_n}, k={gate_up_k}, dtype={self.dtype}, flag=gate_up_proj", 
            f"m={m}, n={down_n}, k={down_k}, dtype={self.dtype}, flag=down_proj", 
        ]
        if self.moe_intermediate_size:
            num_experts = self.config['num_experts']
            results.append(
                f"m={m}, n={num_experts}, k={self.hidden_size}, dtype={self.dtype}, flag=experts_selection"
            )
        return sorted(list(set(results)))
    
    def extract_output_gemm(self, m=1, tp_size=1):
        vocab_size = divide_and_check_no_remainder(self.config['vocab_size'], tp_size)
        results = [
            f"m={m}, n={vocab_size}, k={self.hidden_size}, dtype={self.dtype}, flag=output", 
        ]
        return results
    
    def extract_attention(self, batch_size=1, seq_len=1, tp_size=1):
        nhead_q = self.config['num_attention_heads']
        nhead_kv = self.config['num_key_value_heads']
        head_dim = self.head_dim
        results = [
            f"batch_size={batch_size}, seq_len={seq_len}, nhead_q={nhead_q}, nhead_kv={nhead_kv}, head_dim={head_dim}, dtype={self.dtype}", 
        ]
        return results


if __name__ == '__main__':
    hf_model_dir = sys.argv[1]
    print(hf_model_dir)
    eo = ExtractOps(hf_model_dir)

    attention_gemms = eo.extract_attention_gemm()
    print("attention_gemms")
    for gemm in attention_gemms:
        print(gemm)

    ffn_gemms = eo.extract_ffn_gemm()
    print("ffn_gemms")
    for gemm in ffn_gemms:
        print(gemm)
    
    output_gemms = eo.extract_output_gemm()
    print("output_gemms")
    for gemm in output_gemms:
        print(gemm)
    
    attentions = eo.extract_attention()
    print("attentions")
    for attention in attentions:
        print(attention)
