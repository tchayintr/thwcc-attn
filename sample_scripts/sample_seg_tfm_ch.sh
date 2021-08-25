## Common parameters
set -e

GPU_ID=1
TASK=seg
TAGGING_UNIT=transformer
BATCH=50
ACCUMULATE_GRAD=1


################################################
# Train a segmentation model

MODE=train
EP_BEGIN=1
EP_END=5
BREAK_POINT=100
CHAR_EMB_SIZE=128
TFM_LAYERS=6
TFM_HEADS=8
TFM_FF_HIDDEN_SIZE=2048
TFM_HIDDEN_SIZE=512
MLP_LAYERS=2
TFM_DROPOUT=0.1
MAX_SEQ_LEN=5000
TRAIN_DATA=data/samples/best2010_sample_100.seg.sl
VALID_DATA=data/samples/best2010_sample_20.seg.sl
INFERENCE_LAYER=crf
INPUT_FORMAT=sl

python3 src/segmenter.py \
       --task $TASK \
       --tagging_unit $TAGGING_UNIT \
       --execute_mode $MODE \
       --gpu $GPU_ID \
       --epoch_begin $EP_BEGIN \
       --epoch_end $EP_END \
       --break_point $BREAK_POINT \
       --batch_size $BATCH \
       --accumulate_grad_batches $ACCUMULATE_GRAD \
       --unigram_embed_size $CHAR_EMB_SIZE \
       --tfm_n_layers $TFM_LAYERS \
       --tfm_ff_hidden_size $TFM_FF_HIDDEN_SIZE \
       --tfm_hidden_size $TFM_HIDDEN_SIZE \
       --tfm_n_heads $TFM_HEADS \
       --mlp_n_layers $MLP_LAYERS \
       --tfm_dropout $TFM_DROPOUT \
       --max_seq_len $MAX_SEQ_LEN \
       --inference_layer $INFERENCE_LAYER \
       --train_data $TRAIN_DATA \
       --valid_data $VALID_DATA \
       --input_data_format $INPUT_FORMAT \
       # --quiet \

# MODEL=models/main/yyyymmdd_hhmm_ex.yyy.pt

################################################
# Segment a raw text by the learned model

MODE=decode
DECODE_DATA=data/samples/best2010_sample_10.raw.sl
OUTPUT_DATA=decode_tfm_ch.sl
OUTPUT_FORMAT=sl

# python3 src/segmenter.py \
#        --task $TASK \
#        --execute_mode $MODE \
#        --gpu $GPU_ID \
#        --batch_size $BATCH \
#        --tagging_unit $TAGGING_UNIT \
#        --decode_data $DECODE_DATA \
#        --model_path $MODEL \
#        --output_data_format $OUTPUT_FORMAT \
#        --output_data $OUTPUT_DATA \
#        # --quiet \
