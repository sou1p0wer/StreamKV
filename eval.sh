# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=1

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=llava_ov_7b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie streamingbench ovbench
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
dataset=streamingbench

#****************************segment config****************************
segment_mode=semantic    #(uniform、semantic)
min_chunk_size=4
max_chunk_size=64
segment_theta=0.99
chunk_global_sign=True   #(True、False)
#****************************encode config****************************
n_local=15000
#****************************compress config****************************
compress_mode=online    #(base、offline、online)
compress_temp=0.75
compression_ratio=0.6
encode_mode=dense       #(sparse、dense)only works when compression_ratio < 1
#****************************retrieval config****************************
retrieval_mode=online    #(base、offline、online)
retrieve_temp=0.25
retrieve_size=64
retrieve_local=False     #(True、False)
retrieve_local_size=8

python -m video_qa.run_eval \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --segment_mode ${segment_mode} \
    --min_chunk_size $min_chunk_size \
    --max_chunk_size $max_chunk_size \
    --segment_theta $segment_theta \
    --chunk_global_sign ${chunk_global_sign} \
    --n_local $n_local \
    --compress_mode ${compress_mode} \
    --compress_temp $compress_temp \
    --compression_ratio $compression_ratio \
    --encode_mode ${encode_mode} \
    --retrieval_mode ${retrieval_mode} \
    --retrieve_temp $retrieve_temp \
    --retrieve_size $retrieve_size \
    --retrieve_local ${retrieve_local} \
    --retrieve_local_size $retrieve_local_size
