#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=2
method=ndpp
num_ice=50
port=9927

model_name=EleutherAI/gpt-neo-2.7B
n_tokens=1600
scr_batch_size=8
inf_batch_size=8

for task_name in geoquery mtop webqs nl2bash rocending
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  ceil_model=output/ceil/${task_name}/${model_name}

  retrieve_file=${run_dir}/retrieved.json
  python retriever.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      task_name=${task_name} \
      dataset_reader.dataset_split=train \
      +dataset_reader.ds_size=44000 \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${ceil_model} \
      model_config.norm_embed=true \
      faiss_index=${run_dir}/index \
      ndpp_search=true \
      ndpp_topk=100 \
      num_ice=16 \
      num_candidates=50 \
      model_config.scale_factor=0.1 \
      mode=cand_random

  scored_file=${run_dir}/scored.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  scorer.py \
      hydra.run.dir=${run_dir}/scorer \
      task_name=${task_name} \
      output_file=${scored_file} \
      batch_size=${scr_batch_size} \
      model_name=${model_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data}

  python NDPP-learning/main.py --dataset_name ${task_name} --input_file ${scored_file}

  retrieve_file=${run_dir}/train_retrieved.json
  python retriever.py \
      hydra.run.dir=${run_dir}/retriever \
      output_file=${retrieve_file} \
      num_ice=${num_ice} \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${ceil_model} \
      faiss_index=${run_dir}/index \
      model_config.norm_embed=true \
      model_config.scale_factor=${scale_factor} \
      ndpp_search=true \
      ndpp_topk=100 \
      mode=map

echo "retrieval done"
  pred_file=${run_dir}/pred.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size}
echo "inferencer done"
done