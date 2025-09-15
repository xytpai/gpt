import os
import re
import sys
import glob
from safetensors.torch import load_file


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class ExtractOps:
    def __init__(self, hf_model_dir):
        self.load_state_dict(hf_model_dir) 
        self.extract_state_dict_meta()
    
    def load_state_dict(self, hf_model_dir):
        files = glob.glob(os.path.join(hf_model_dir, "*.safetensors"))
        state_dict = {}
        for file in files:
            state_dict.update(load_file(file, device='cpu'))
        self.state_dict = state_dict
    
    def extract_state_dict_meta(self):
        new_state_dict = {}
        for k, v in self.state_dict.items():
            new_state_dict[k] = v.to('meta')
        self.state_dict = new_state_dict
    
    def extract_attention_gemm(self, m=1, tp_size=1):
        results = []
        hidden_size = None
        for key, meta in self.state_dict.items():
            if ('.q_proj.weight' in key) or ('.k_proj.weight' in key) or ('.v_proj.weight' in key):
                n = divide_and_check_no_remainder(meta.shape[0], tp_size)
                k = meta.shape[1]
                if not hidden_size:
                    hidden_size = k
                assert hidden_size == k
                results.append(f"m={m}, n={n}, k={k}, dtype={meta.dtype}")
            if '.o_proj.weight' in key:
                n = meta.shape[0]
                k = divide_and_check_no_remainder(meta.shape[1], tp_size)
                if not hidden_size:
                    hidden_size = n
                assert hidden_size == n
                results.append(f"m={m}, n={n}, k={k}, dtype={meta.dtype}")
        results = list(set(results))
        return sorted(results)
    
    def extract_ffn_gemm(self, m=1, tp_size=1):
        results = []
        hidden_size = None
        for key, meta in self.state_dict.items():
            if ('.gate_proj.weight' in key) or ('.up_proj.weight' in key):
                n = divide_and_check_no_remainder(meta.shape[0], tp_size)
                k = meta.shape[1]
                if not hidden_size:
                    hidden_size = k
                assert hidden_size == k
                results.append(f"m={m}, n={n}, k={k}, dtype={meta.dtype}")
            if '.down_proj.weight' in key:
                n = meta.shape[0]
                k = divide_and_check_no_remainder(meta.shape[1], tp_size)
                if not hidden_size:
                    hidden_size = n
                assert hidden_size == n
                results.append(f"m={m}, n={n}, k={k}, dtype={meta.dtype}")
        results = list(set(results))
        return sorted(results)
    
    def extract_output_gemm(self, m=1, tp_size=1):
        results = []
        for key, meta in self.state_dict.items():
            if 'output.weight' in key:
                n = divide_and_check_no_remainder(meta.shape[0], tp_size)
                k = meta.shape[1]
                results.append(f"m={m}, n={n}, k={k}, dtype={meta.dtype}")
        results = list(set(results))
        return sorted(results)
    
    def print_gemms(self, results):
        for s in results:
            m, n, k, _ = map(int, re.findall(r"\d+", s))
            print(m, n, k)


if __name__ == '__main__':
    hf_model_dir = sys.argv[1]
    eo = ExtractOps(hf_model_dir)
    attention_gemms = eo.extract_attention_gemm()
    ffn_gemms = eo.extract_ffn_gemm()
    output_gemms = eo.extract_output_gemm()
    all_gemms = sorted(set(attention_gemms + ffn_gemms + output_gemms))
    print(hf_model_dir, "all_gemms: m,n,k,dtype")
    for gemm in all_gemms:
        print(gemm)

    print(' ')
    print('attention_gemms')
    eo.print_gemms(attention_gemms)
    print('ffn_gemms')
    eo.print_gemms(ffn_gemms)
    print('output_gemms')
    eo.print_gemms(output_gemms)
