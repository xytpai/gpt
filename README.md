This is a minimalist inference project for transformer based language models, currently supporting the following models:

- Llama3.2-1B-Instruct
- Llama3.2-3B-Instruct
- Llama3.3-70B-Instruct
- Qwen3-4B-Instruct

#### Example

Download weights from https://www.llama.com/llama-downloads. Then, modify **wfiles** and **tfile** in the corresponding json file to specify the locations of model weights and vocab-embedding weights. The running CMD:

```bash
python modeling.py models/Llama3.2-1B-Instruct.json "What's the tallest mountain?"
```

Outputs:

```txt
The tallest mountain in the world is Mount Everest, located in the Himalayas on the border between Nepal and Tibet, China. It stands at an impressive 8,848.86 meters (29,031.7 feet) above sea level.

However, if you're asking about the tallest mountain when measured from its base to its summit, then Mount Everest is often referred to as the tallest mountain in the world.

But if you're asking about the tallest mountain when measured from its base to its lowest point to the sea level, then the answer is Mount Kailash in Tibet, with a height of 7, dipping down to around 3,100 meters (10,200 feet) below sea level.

But Mount Everest is generally considered the tallest mountain in the world!
```

#### Huggingface Scripts

Simple CMD for inference using huggingface API:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ~/data/qwen3-4b/
python infer_hf.py ~/data/qwen3-4b/ "Introduce yourself in Chinese"
```
