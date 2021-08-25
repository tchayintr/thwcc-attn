## Common parameters
set -e

GPU_ID=-1
TASK=seg
TAGGING_UNIT=mutant
BATCH=50
CC_EMBED_MODEL= #/path/to/cc_embedding_model


################################################
# Train a segmentation model

MODE=train
EP_BEGIN=1
EP_END=5
BREAK_POINT=100
CHAR_EMB_SIZE=128
CC_EMB_SIZE=300
RNN_LAYERS=2
RNN_HIDDEN_SIZE=600
RNN2_LAYERS=2
RNN2_HIDDEN_SIZE=600
RNN_DROPOUT=0.4
CC_VEC_DROPOUT=0.4
COMPOSITION_TYPE=cccon
TRAIN_DATA=data/samples/best2010_sample_100.seg.sl
VALID_DATA=data/samples/best2010_sample_20.seg.sl
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
       --unigram_embed_size $CHAR_EMB_SIZE \
       --cc_embed_size $CC_EMB_SIZE \
       --rnn_bidirection \
       --rnn_n_layers $RNN_LAYERS \
       --rnn_hidden_size $RNN_HIDDEN_SIZE \
       --rnn_n_layers2 $RNN2_LAYERS \
       --rnn_hidden_size2 $RNN2_HIDDEN_SIZE \
       --rnn_dropout $RNN_DROPOUT \
       --cc_vector_dropout $CC_VEC_DROPOUT \
       --pretrained_embed_usage init \
       --cc_pooling_type $COMPOSITION_TYPE \
       --train_data $TRAIN_DATA \
       --valid_data $VALID_DATA \
       --input_data_format $INPUT_FORMAT \
       --rnn_batch_first \
       --use_gold_cc \
       # --model_path $MODEL \
       # --cc_embed_model_path $CC_EMBED_MODEL \

# MODEL=models/main/yyyymmdd_hhmm_ex.yyy.pt

################################################
# Segment a raw text by the learned model

MODE=decode
DECODE_DATA=data/samples/best2010_sample_10.raw.sl
OUTPUT_DATA=decode_cc_cccon.sl
OUTPUT_FORMAT=sl

# python3 src/segmenter.py \
#        --task $TASK \
#        --tagging_unit $TAGGING_UNIT \
#        --execute_mode $MODE \
#        --gpu $GPU_ID \
#        --batch_size $BATCH \
#        --decode_data $DECODE_DATA \
#        --model_path $MODEL \
#        --output_data_format $OUTPUT_FORMAT \
#        --quiet \
#        --output_data $OUTPUT_DATA \
#         # --cc_embed_model_path $CC_EMBED_MODEL \
