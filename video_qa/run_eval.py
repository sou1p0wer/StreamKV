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

def eval_streamingbench_omni(args):
    num_chunks = args.num_chunks
    save_dir = f"results/streamingbench/omni-{args.segment_mode}-{args.min_segment_size}-{args.max_segment_size}-{args.segment_theta}-{args.use_segment_summary}-{args.n_local}-{args.compress_temp}-{args.compression_ratio}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "streamkv_stream_vqa"
    anno = "streamingbench/question_omni_streamkv_online.json"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", os.path.join(args.data_root, anno),
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_temp", str(args.compress_temp),
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_segment_size", str(args.max_segment_size),
                    "--min_segment_size", str(args.min_segment_size),
                    "--segment_theta", str(args.segment_theta),
                    "--use_segment_summary", args.use_segment_summary,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_streamingbench_real(args):
    num_chunks = args.num_chunks
    save_dir = f"results/streamingbench/real-{args.segment_mode}-{args.min_segment_size}-{args.max_segment_size}-{args.segment_theta}-{args.use_segment_summary}-{args.n_local}-{args.compress_temp}-{args.compression_ratio}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "streamkv_stream_vqa"
    anno = "streamingbench/question_real_streamkv_online.json"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", os.path.join(args.data_root, anno),
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_temp", str(args.compress_temp),
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_segment_size", str(args.max_segment_size),
                    "--min_segment_size", str(args.min_segment_size),
                    "--segment_theta", str(args.segment_theta),
                    "--use_segment_summary", args.use_segment_summary,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_streamingbench_sqa(args):
    num_chunks = args.num_chunks
    save_dir = f"results/streamingbench/sqa-{args.segment_mode}-{args.min_segment_size}-{args.max_segment_size}-{args.segment_theta}-{args.use_segment_summary}-{args.n_local}-{args.compress_temp}-{args.compression_ratio}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "streamkv_stream_vqa"
    anno = "streamingbench/question_sqa_streamkv_online.json"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", os.path.join(args.data_root, anno),
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_temp", str(args.compress_temp),
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_segment_size", str(args.max_segment_size),
                    "--min_segment_size", str(args.min_segment_size),
                    "--segment_theta", str(args.segment_theta),
                    "--use_segment_summary", args.use_segment_summary,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))  # llava_ov_72b needs 4x 80GB GPUs
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

def eval_streamingbench_proactive(args):
    num_chunks = args.num_chunks
    save_dir = f"results/streamingbench/proactive-{args.segment_mode}-{args.min_segment_size}-{args.max_segment_size}-{args.segment_theta}-{args.use_segment_summary}-{args.n_local}-{args.compress_temp}-{args.compression_ratio}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "streamkv_proactive_vqa"
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
                    "--anno_path", os.path.join(args.data_root, "streamingbench/question_proactive_streamkv_online.json"),
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_temp", str(args.compress_temp),
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_segment_size", str(args.max_segment_size),
                    "--min_segment_size", str(args.min_segment_size),
                    "--segment_theta", str(args.segment_theta),
                    "--use_segment_summary", args.use_segment_summary,
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
    exec(f"python video_qa/eval/eval_proactive.py --save_dir {save_dir}")

def eval_videomme(args):
    num_chunks = args.num_chunks
    save_dir = f"results/videomme/{args.segment_mode}-{args.min_segment_size}-{args.max_segment_size}-{args.segment_theta}-{args.use_segment_summary}-{args.n_local}-{args.compress_temp}-{args.compression_ratio}-{args.retrieve_temp}-{args.retrieve_size}-{args.retrieve_local}-{args.retrieve_local_size}"
    solver = "streamkv_offline_vqa"
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", f"video_qa/{solver}.py",
                    "--model", args.model,
                    "--sample_fps", str(args.sample_fps),
                    "--n_local", str(args.n_local),
                    "--retrieve_size", str(args.retrieve_size),
                    "--save_dir", save_dir,
                    "--anno_path", os.path.join(args.data_root, "videomme/data.json"),
                    "--debug", args.debug,
                    "--num_chunks", str(num_chunks),
                    "--dataset", args.dataset,
                    "--compression_ratio", str(args.compression_ratio),
                    "--compress_temp", str(args.compress_temp),
                    "--retrieve_temp", str(args.retrieve_temp),
                    "--retrieve_local", args.retrieve_local,
                    "--retrieve_local_size", str(args.retrieve_local_size),
                    "--max_segment_size", str(args.max_segment_size),
                    "--min_segment_size", str(args.min_segment_size),
                    "--segment_theta", str(args.segment_theta),
                    "--use_segment_summary", args.use_segment_summary,
                    "--segment_mode", args.segment_mode,
                    "--chunk_idx", str(idx)]
            p = multiprocessing.Process(target=exec, args=(cmd, True, f'{4*idx},{4*idx+1},{4*idx+2},,{4*idx+3}' if args.model=='llava_ov_72b' else str(idx)))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        exec(f"> {save_dir}/results.csv")
        for idx in range(num_chunks):
            if idx == 0:
                exec(f"head -n 1 {save_dir}/{num_chunks}_{idx}.csv > {save_dir}/results.csv")
            exec(f"tail -n +2 {save_dir}/{num_chunks}_{idx}.csv >> {save_dir}/results.csv")
            exec(f"rm {save_dir}/{num_chunks}_{idx}.csv")
    exec(f"python video_qa/eval/eval_multiple_choice.py --save_dir {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_7b", choices=['llava_ov_0.5b', 'llava_ov_7b', 'llava_ov_72b'])
    parser.add_argument("--dataset", type=str, default=None, choices=['streamingbench_omni', 'streamingbench_real', 'streamingbench_sqa', 'videomme', 'streamingbench_proactive'])
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--compression_ratio", type=float, default=1)
    parser.add_argument("--compress_temp", type=float, default=0)
    parser.add_argument("--retrieve_temp", type=float, default=0)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_local", type=str, default='True', choices=['True', 'False'])
    parser.add_argument("--retrieve_local_size", type=int, default=8)
    parser.add_argument("--segment_mode", type=str, default='uniform', choices=['uniform', 'semantic'])
    parser.add_argument("--max_segment_size", type=int, default=64)
    parser.add_argument("--min_segment_size", type=int, default=8)
    parser.add_argument("--segment_theta", type=float, default=0.95)
    parser.add_argument("--use_segment_summary", type=str, default='True', choices=['True', 'False'])
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()
    func_dic = {
        'streamingbench_omni': eval_streamingbench_omni,
        'streamingbench_real': eval_streamingbench_real,
        'streamingbench_sqa': eval_streamingbench_sqa,
        'videomme': eval_videomme,
        'streamingbench_proactive': eval_streamingbench_proactive,
    }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
