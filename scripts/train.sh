#!/usr/bin/env bash
export PYTHONPATH=/home/***/lib/apex:/home/***/lib/cocoapi:/home/***/code/scene_graph_gen/scene_graph_benchmark_pytorch:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GUP=1
    echo "TRAINING Predcls"
    dataset="oi"
    MODEL_NAME="transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4_${dataset}" #"transformer_predcls_dist15_3k_FixPModel_lr1e3_B16_FCMat"
    # MODEL_NAME="transformer_predcls_dist15_3k_FixPModel_lr1e3_B16_FCMat"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10050 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer_${dataset}.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupReduceLROnPlateau \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME}
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=1
    export NUM_GUP=1
    echo "TRAINING SGcls"
    MODEL_NAME="transformer_sgcls_dist15_2k_confmat_woInit_second"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10020 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupReduceLROnPlateau \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GUP=1
    echo "TRAINING SGdet"
    MODEL_NAME="transformer_sgdet_dist15_2k_confmat_woInit"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10021 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupReduceLROnPlateau \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "3" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GUP=1
    echo "TRAINING SGcls"
    MODEL_NAME="transformer_sgcls_dist15_2k_confmat_woInit_no_second"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10020 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupReduceLROnPlateau \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "4" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export NUM_GUP=1
    echo "TRAINING Predcls"
    # MODEL_NAME="transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4" #"transformer_predcls_dist15_3k_FixPModel_lr1e3_B16_FCMat"
    MODEL_NAME="transformer_predcls_dist15_3k_FixPModel_lr1e3_B16_FCMat"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10050 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupReduceLROnPlateau \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
    MODEL.PRETRAINED_MODEL_CKPT '' \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME}

elif [ $1 == "5" ]; then
    export CUDA_VISIBLE_DEVICES=0 #3,4 #,4 #3,4
    export NUM_GUP=1
    echo "TRAINING SGdet"
    MODEL_NAME="transformer_sgdet_dist15_2k_confmat_woInit_second"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10021 --nproc_per_node=$NUM_GUP \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    DTYPE "float32" \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
    SOLVER.MAX_ITER 16000 SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupReduceLROnPlateau \
    SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
    SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
    MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
fi
