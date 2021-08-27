# Character-based Thai Word Segmentation with Multiple Attentions
#### _thwcc-attn_

Yet another Thai Word Segmentation that employs multiple linguistic information, including characters, character-cluster, subwords, and words, with attention mechanisms.

### Architecture
- Character-based word segmentation
- BiLSTM-CRF architecture
- Words and Character-cluster/subword attentions integrated to character representations
- BIES tagging scheme
  -  B: beginning, I: inside, E: end, and S: single

### Segmentation Performance (micro-averaged f1 score)
_CWCC-WCON_
- CoNLL (word-level): 97.67
    - [conlleval]( https://github.com/spyysalo/conlleval.py)
    - A simplified word-level evaluation script
        -  https://github.com/tchayintr/word-level-eval.py
- BIES (character-level): 98.99
- Boundary (boundary-level): 99.38
    - http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.858.pdf 

### Datasets (based on BEST2010 corpus)
- train, valid, test 
    - https://resources.aiat.or.th/thwcc-attn/
    - raw.sl: unsegmented-sentence line
    - seg.sl: segmented-sentence line

### Requirements
- python3 >= 3.7.3
- torch >= 1.6.0+cu101 (original: 1.5.0+cu101)
- allennlp >= 1.1.0 (original: 1.0.0)
- numpy >= 1.19.2
- pathlib >= 1.0.1
- gensim >= 3.8.3
- pickle

### Modes
- **train**: train and evaluate a model
- **decode**: decode an input file (unsegmented text) to a segmented words

#### Data format
- **sl**: sentence line

### Usage
_Modes can be specified by executing the following the sample scripts_
#### Training models
- Character-BiLSTM-CRF(baseline): sample_scripts/sample_seg_ch.sh
    - tagging_unit: single
    - ``./sample_scripts/sample_seg_ch.sh``
- Character-Transformer-CRF: sample_scripts/sample_seg_tfm.sh
    - tagging_unit: transformer
    - ``./sample_scripts/sample_seg_tfm_ch.sh``
- W-WCON (strong-baseline): sample_scripts/sample_seg_w.sh
    - tagging_unit: hybrid
    - word-attention
    - ``./sample_scripts/sample_seg_w.sh``
- CCC-WCON (preliminary): 
    - tagging_unit: mutant
    - cc-attention
    - ``./sample_scripts/sample_seg_cc.sh``
- CWSW-WCON (preliminary): sample_scripts/sample_seg_wsw.sh
    - tagging_unit: sub-combinative
    - word-attention and subword-attention
    - ``./sample_scripts/sample_seg_wsw.sh``
- CWCC-WCON (The best model): sample_scripts/sample_seg_wcc.sh
    - tagging_unit: combinative
    - word-attention and cc-attention
    - ``./sample_scripts/sample_seg_wcc.sh``

#### Logs
- A log file will be saved in ``log`` 
    - training/evaluating scores
    - hyperparameters

#### Trained models
- Trained models will be saved in ``models/main``
    - hyperparameters
    - dictionary
    - checkpoint for each break point

### Acknowledgement
-  Implementations based on modification of [seikanlp](https://github.com/shigashiyama/seikanlp)

### Citation
- TBA
