# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=8

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b
model=llava_ov_7b

# Supported dataset: streamingbench_omni streamingbench_real streamingbench_sqa streamingbench_proactive
dataset=streamingbench_omni

#****************************segment config****************************
segment_mode=semantic    #(uniform、semantic)
min_segment_size=4
max_segment_size=64
segment_theta=0.99
use_segment_summary=True   #(True、False)
#****************************encode config****************************
n_local=15000
#****************************compress config****************************
compress_temp=1
compression_ratio=0.6
#****************************retrieval config****************************
retrieve_temp=0.5
retrieve_size=64
retrieve_local=True     #(True、False)
retrieve_local_size=8

python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --segment_mode ${segment_mode} \
    --min_segment_size $min_segment_size \
    --max_segment_size $max_segment_size \
    --segment_theta $segment_theta \
    --use_segment_summary ${use_segment_summary} \
    --n_local $n_local \
    --compress_temp $compress_temp \
    --compression_ratio $compression_ratio \
    --retrieve_temp $retrieve_temp \
    --retrieve_size $retrieve_size \
    --retrieve_local ${retrieve_local} \
    --retrieve_local_size $retrieve_local_size
