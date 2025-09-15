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
    
    def extract_attention_gemm(self, m=1, tp_size=1):
        q_n = self.config['head_dim'] * self.config['num_attention_heads']
        q_n = divide_and_check_no_remainder(q_n, tp_size)
        kv_n = self.config['head_dim'] * self.config['num_key_value_heads']
        kv_n = divide_and_check_no_remainder(kv_n, tp_size)
        qkv_k = self.hidden_size
        results = [
            f"m={m}, n={q_n}, k={qkv_k}, dtype={self.dtype}", # q
            f"m={m}, n={kv_n}, k={qkv_k}, dtype={self.dtype}", # kv
            f"m={m}, n={qkv_k}, k={q_n}, dtype={self.dtype}", # o
        ]
        return sorted(list(set(results)))

    def extract_ffn_gemm(self, m=1, tp_size=1):
        gate_up_n = divide_and_check_no_remainder(self.config['intermediate_size'], tp_size)
        gate_up_k = self.hidden_size
        down_n = gate_up_k
        down_k = gate_up_n
        results = [
            f"m={m}, n={gate_up_n}, k={gate_up_k}, dtype={self.dtype}", 
            f"m={m}, n={down_n}, k={down_k}, dtype={self.dtype}", 
        ]
        return sorted(list(set(results)))
    
    def extract_output_gemm(self, m=1, tp_size=1):
        vocab_size = divide_and_check_no_remainder(self.config['vocab_size'], tp_size)
        results = [
            f"m={m}, n={vocab_size}, k={self.hidden_size}, dtype={self.dtype}", 
        ]
        return results
    
    def extract_attention(self, batch_size=1, seq_len=1, tp_size=1):
        nhead_q = self.config['num_attention_heads']
        nhead_kv = self.config['num_key_value_heads']
        head_dim = self.config['head_dim']
        results = [
            f"batch_size={batch_size}, seq_len={seq_len}, nhead_q={nhead_q}, nhead_kv={nhead_kv}, head_dim={head_dim}, dtype={self.dtype}", 
        ]
        return results


if __name__ == '__main__':
    hf_model_dir = sys.argv[1]
    eo = ExtractOps(hf_model_dir)
    attention_gemms = eo.extract_attention_gemm()
    ffn_gemms = eo.extract_ffn_gemm()
    output_gemms = eo.extract_output_gemm()
    all_gemms = sorted(set(attention_gemms + ffn_gemms + output_gemms))
    attentions = eo.extract_attention()
    print(hf_model_dir)
    print("all_gemms")
    for gemm in all_gemms:
        print(gemm)
    print("attentions")
    for attention in attentions:
        print(attention)
