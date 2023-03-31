### GPT

This is the simplest natural language processing model. 
The entire project includes distributed training and inference.
The training method is to predict the next word.
Please make sure that you have installed a CUDA environment with pytorch >=1.12. The specific usage instructions are as follows:

#### Setup

```bash
pip install -r requirements.txt
```

#### Training

```bash
python train.py --end=40000
```

#### Predicting

```bash
python eval.py --text=Hello
```
