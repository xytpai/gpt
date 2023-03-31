### GPT

This is the simplest natural language processing model. 
The entire project includes distributed training and inference.
The training method is to predict the next word.
Please make sure that you have installed a CUDA environment with pytorch >=1.12. The specific usage instructions are as follows:

#### Setup

```bash
pip install -r requirements.txt
```

#### Get dataset (Wiki)

```bash
export WIKI_ROOT=/data/wiki/ # Change to your own path
rm -rf ${WIKI_ROOT}
python ./data/wiki/wiki_downloader.py --language=en --save_path=${WIKI_ROOT}
# Or download from: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# bzip2 -dk ${WIKI_ROOT}enwiki*
WIKI_FILE=$(find /data/wiki/ -type f -iname enwiki*.xml)
python ./data/wiki/WikiExtractor.py ${WIKI_FILE} -o ${WIKI_ROOT}text
cd data/wiki/wikicleaner/
bash run.sh "${WIKI_ROOT}text/*/wiki_??" ${WIKI_ROOT}results
```

#### Training

```bash
python train.py       \
    --model=nano      \
    --batch_size=8    \
    --data=./minidata \
    --end=40000
```

#### Predicting

```bash
python eval.py --model=nano --text=Hello
```
