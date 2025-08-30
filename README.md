This is a minimalist inference project for transformer based language models, currently supporting the following models:

#### Llama

Download weights from https://www.llama.com/llama-downloads. Then, modify **wfiles** and **tfile** in the corresponding json file to specify the locations of model weights and vocab-embedding weights. The running CMD:

```bash
python modeling.py models/Llama3.2-1B-Instruct.json "What's the tallest mountain?"

```

Example outputs:

```txt
<|start_header_id|>assistant<|end_header_id|>

The tallest mountain is Mount Everest, located in the Himalayas on the border between Nepal and Tibet, China. It stands at an impressive 8,848 meters (29,029 feet) above sea level.
```

#### Huggingface Scripts

Simple CMD for inference using huggingface API:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 ~/data/qwen3-4b/
python infer_hf.py ~/data/qwen3-4b/ "Introduce yourself in Chinese"
```
