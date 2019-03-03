---
title: Model Training and Evaluation
keywords: stanfordnlp, model training
permalink: '/training.html'
---

## Overview

All neural modules, including the tokenzier, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer and the dependency parser, can be trained with your own [CoNLL-U](https://universaldependencies.org/format.html) format data. Currently, we do not support model training via the `Pipeline` interface. Therefore, to train your own models, you will need to clone the source code from the [git repository](https://github.com/stanfordnlp/stanfordnlp) and follow the procedures below.

If you only want to run the processors with the pretrained models, please skip this and go to [the Pipeline page](pipeline.md).


## Setting Environment Variables

We provide scripts that are useful for model training and evaluation in the `scripts` folder. The first thing to do before training is to setup the environment variables by editing `scripts/config.sh`. The most important variables include:
- `UDBASE`: This should point to the root directory of your training/dev/test `CoNLL-U` data. For examples of this folder, you can download the raw data from the [CoNLL 2018 UD Shared Task website](http://universaldependencies.org/conll18/data.html).
- `DATA_ROOT`: This is the root directory for storing intermediate training files generated by the scripts.
- `{module}_DATA_DIR`: The subdirectory for storing intermediate files used by each module.
- `WORDVEC_DIR`: The directory to store all word vector files (see below).


## Preparing Word Vector Data

To train modules that make use of word representations, such as the tagger and dependency parser, it is highly recommended that you use pretrained embedding vectors. To replicate the system performance on the CONLL 2018 shared task, we have prepared a script for you to download all word vector files. Simply run from the source directory:
```bash
bash scripts/download_vectors.sh ${wordvec_dir}
```
where `${wordvec_dir}` is the target directory to store the word vector files, and should be the same as where the environment variable `WORDVEC_DIR` is pointed to. 

The above script will first download the pretrained word2vec embeddings released from the CoNLL 2017 Shared Task, which can be found [here](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar?sequence=9&isAllowed=y). For languages not in this list, it will download the [FastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) from Facebook. Note that the total size of all downloaded vector files will be ~30G, therefore please use this script with caution.

After running the script, your embedding vector files will be organized in the following way:
`${WORDVEC_DIR}/{language}/{language_code}.vectors.xz`. For example, the word2vec file for English should be put into `$WORDVEC_DIR/English/en.vectors.xz`. If you use your own vector files, please make sure you arrange them in a similar fashion as described above.

## Training with Scripts

We provide various bash scripts to ease the training process in the `scripts` directory. To train a model, you can run the following command from the code root directory:
```bash
bash scripts/run_${module}.sh ${treebank} ${other_args}
```
where `${module}` is one of `tokenize`, `mwt`, `pos`, `lemma` or `depparse`; `${treebank}` is the full name of the treebank; `${other_args}` are other arguments allowed by the training script.

For example, you can use the following command to train a tokenizer with batch size 32 and a dropout rate of 0.33 on the `UD_English-EWT` treebank:

```bash
bash scripts/run_tokenize.sh UD_English-EWT --batch_size 32 --dropout 0.33
```

Note that for the dependency parser, you also need to specify `gold|predicted` for the used POS tag type in the training/dev data.
```bash
bash scripts/run_depparse.sh UD_English-EWT gold
```
If `predicted` is used, the trained tagger model will first be run on the training/dev data to generate the predicted tags.


For a full list of available training arguments, please refer to the specific entry point of that module. By default model files will be saved to the `saved_models` directory during training (which can also be changed with the `save_dir` argument).


## Evaluation

Model evaluation will be run automatically after each training run. Additionally, after you finish training all modules, you can evaluate the full end-to-end system with this command:
```bash
bash scripts/run_ete.sh ${treebank} ${split}
```
where `${split}` is one of train, dev or test.


## Devices

We **strongly encourage** you to train all modules with a **GPU** device. When a CUDA device is available and detected by the script, the CUDA device will be used automatically; otherwise, CPU will be used. However, you can force the training to happen on a CPU device, by specifying `--cpu` when calling the script.