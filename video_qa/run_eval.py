import os
import argparse
import subprocess
import multiprocessing


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)

def eval_ovbench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/ovobench/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "/home/cyl476530/ReKV/data/OVOBench/OVO-Bench/data/data.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            #p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx % args.num_gpus)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


def eval_streambench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/streambench/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "/home/cyl476530/ReKV/data/StreamBench/streambench.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_streamingbench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/streamingbench/omni-{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "/home/cyl476530/ReKV/data/StreamingBench/src/data/question_omni_rekv_online.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            #p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx % args.num_gpus)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_lvbench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/lvbench/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "/home/cyl476530/ReKV/data/lvbench/data.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_videomme(args):
    num_chunks = args.num_chunks
    save_dir = f"results/videomme/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "/home/cyl476530/ReKV/data/videomme/data.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_qaego4d(args):
    num_chunks = args.num_chunks
    save_dir = f"results/qaego4d/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "/home/cyl476530/ReKV/data/qaego4d/test_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_egoschema(args):
    num_chunks = args.num_chunks
    save_dir = f"results/egoschema/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/egoschema/full.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_egoschema.py --save_dir {save_dir}")

def eval_activitynet_qa(args):
    num_chunks = args.num_chunks
    save_dir = f"results/activitynet_qa/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/activitynet_qa/test.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_ego(args):
    num_chunks = args.num_chunks
    save_dir = f"results/rvs_ego/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/ego/ego4d_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_rvs_movie(args):
    num_chunks = args.num_chunks
    save_dir = f"results/rvs_movie/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_stream_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/rvs/movie/movienet_oe.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        exec(f"rm -rf {save_dir}/tmp")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_open_ended.py --pred_path {save_dir}/results.csv --output_dir {save_dir}/tmp --output_json {save_dir}/results.json")

def eval_cgbench(args):
    num_chunks = args.num_chunks
    save_dir = f"results/cgbench/{args.segment_mode}-{args.min_chunk_size}-{args.max_chunk_size}-{args.segment_theta}-{args.chunk_global_sign}-{args.n_local}-{args.compress_mode}-{args.compress_temp}-{args.compression_ratio}-{args.encode_mode}-{args.retrieval_mode}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "rekv_offline_vqa"
    if not args.only_eval:
        # QA
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", "data/cgbench/full_mc.json",
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_mode", args.compress_mode,
                    "--compress_temp", str(args.compress_temp),
                    "--encode_mode", args.encode_mode, 
                    "--retrieval_mode", args.retrieval_mode,
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_chunk_size", str(args.max_chunk_size),
                    "--min_chunk_size", str(args.min_chunk_size),
                    "--segment_theta", str(args.segment_theta),
                    "--chunk_global_sign", args.chunk_global_sign,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # merge results
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    # eval
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b', 'video_llava_7b', 'longva_7b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench', 'streamingbench', 'streambench', 'ovbench', 'videomme','lvbench'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000) #local window size
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--compression_ratio", type=float, default=1)
    parser.add_argument("--compress_mode", type=str, default='base', choices=['offline', 'online', 'base'])
    parser.add_argument("--compress_temp", type=float, default=0)
    parser.add_argument("--encode_mode", type=str, default='dense', choices=['sparse', 'dense'])
    parser.add_argument("--retrieval_mode", type=str, default='base', choices=['offline', 'online', 'base'])
    parser.add_argument("--retrieve_temp", type=float, default=0)
    parser.add_argument("--retrieve_size", type=int, default=64) #the number of retrieved frames
    parser.add_argument("--retrieve_local", type=str, default='True', choices=['True', 'False'])
    parser.add_argument("--retrieve_local_size", type=int, default=8)
    parser.add_argument("--segment_mode", type=str, default='uniform', choices=['uniform', 'semantic'])
    parser.add_argument("--max_chunk_size", type=int, default=64)
    parser.add_argument("--min_chunk_size", type=int, default=8)
    parser.add_argument("--segment_theta", type=float, default=0.95)
    parser.add_argument("--chunk_global_sign", type=str, default='True', choices=['True', 'False'])
    args = parser.parse_args()
    func_dic = {
        'qaego4d': eval_qaego4d,
        'egoschema': eval_egoschema,
        'activitynet_qa': eval_activitynet_qa,
        'rvs_ego': eval_rvs_ego,
        'rvs_movie': eval_rvs_movie,
        'cgbench': eval_cgbench,
        'streamingbench': eval_streamingbench,
        'streambench': eval_streambench,
        'ovbench': eval_ovbench,
        'videomme':eval_videomme,
        'lvbench':eval_lvbench
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
