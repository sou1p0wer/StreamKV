import warnings
import random
import json
import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    logging,
    LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor,
    VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import logzero
from logzero import logger

from model import llava_onevision_rekv, video_llava_rekv, longva_rekv
import av


MODELS = {
    'llava_ov_0.5b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    },
    'llava_ov_7b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
    },
    'llava_ov_72b': {
        'load_func': llava_onevision_rekv.load_model,
        'model_class': LlavaOnevisionForConditionalGeneration,
        'processor_class': LlavaOnevisionProcessor,
        'model_path': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    },
    'video_llava_7b': {
        'load_func': video_llava_rekv.load_model,
        'model_class': VideoLlavaForConditionalGeneration,
        'processor_class': VideoLlavaProcessor,
        'model_path': 'model_zoo/Video-LLaVA-7B-hf',
    },
    'longva_7b': {
        'load_func': longva_rekv.load_model,
        'model_path': 'model_zoo/LongVA-7B',
    },
}


class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps,
                 qa_model, qa_processor=None,
                 num_chunks=None, chunk_idx=None,
                 retrieve_size=64, chunk_size=1) -> None:
        
        self.sample_fps = sample_fps

        self.qa_model = qa_model
        self.qa_processor = qa_processor

        # Retrieval Hyperparams
        assert chunk_size <= retrieve_size, f'chunk_size: {chunk_size}, retrieve_size: {retrieve_size}'
        self.retrieve_size = retrieve_size
        self.chunk_size = chunk_size

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        #print(f'num_chunks: {num_chunks}, chunk_idx: {chunk_idx}')
        if num_chunks is not None:
            anno = self.get_chunk(anno, num_chunks, chunk_idx)
        self.anno = anno
        self.eval_grounding = 'temporal_windows' in anno[0]['conversations'][0]

        self.save_dir = save_dir
        self.choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.record = {(self.retrieve_size, self.chunk_size): []}

        self.performance_records = []
        self.all_ttfts = []
        self.all_throughputs = []
        self.qa_e2e_times = []
        self.qa_records = []

    def split_list(self, lst, n):
        """Split a list into n (roughly) equal-sized chunks"""
        chunk_size = math.ceil(len(lst) / n)  # integer division
        #print(f'len_lst: {len(lst)},chunk_size: {chunk_size}')
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(self, lst, n, k):
        chunks = self.split_list(lst, n)
        #print(f'len(chunks): {len(chunks)},chunk_idx: {k}')
        return chunks[k]

    def load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f'video shape: {video.shape}')
        return video
    
    def summarize_performance(self):
        """
        在所有视频处理完毕后，计算、打印并保存详细的性能统计数据。
        """
        if not self.performance_records and not self.all_ttfts:
            logger.warning("No performance or TTFT records found to summarize.")
            return
        
        summary = {}

        if self.performance_records:
            df = pd.DataFrame(self.performance_records)
            summary['total_videos_processed'] = len(df)
            if 'metrics_e2e_total_time_s' in df.columns:
                e2e_times = df['metrics_e2e_total_time_s']
                summary['time_stats'] = {
                    'avg_s': e2e_times.mean(), 'median_s': e2e_times.median(), 'std_dev_s': e2e_times.std(),
                }
            if 'metrics_peak_memory_gb' in df.columns:
                peak_mems = df['metrics_peak_memory_gb']
                summary['memory_stats_gb'] = {
                    'max': peak_mems.max(), 'p95': peak_mems.quantile(0.95), 'median': peak_mems.median(),
                }
            if 'metrics_avg_encoding_fps_per_qa' in df.columns:
                avg_fps_per_qa_list = df['metrics_avg_encoding_fps_per_qa'].dropna()
                if not avg_fps_per_qa_list.empty:
                    summary['interactive_fps_stats'] = {
                        'overall_avg_fps': avg_fps_per_qa_list.mean(), 'median_of_avg_fps': avg_fps_per_qa_list.median(),
                    }
        
        # <--- 修改4: 添加TTFT的统计逻辑 ---
        if self.all_ttfts:
            summary['ttft_stats_s'] = {
                'total_qa_events': len(self.all_ttfts),
                'mean': np.mean(self.all_ttfts),
                'std_dev': np.std(self.all_ttfts),
                'median': np.median(self.all_ttfts),
                'p95': np.percentile(self.all_ttfts, 95),
            }
        if self.all_throughputs:
            summary['decoding_throughput_stats'] = {
                'mean_tokens_per_sec': np.mean(self.all_throughputs),
                'std_dev_tokens_per_sec': np.std(self.all_throughputs),
                'median_tokens_per_sec': np.median(self.all_throughputs),
            }

        # --- 打印和保存汇总结果 ---
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if 'total_videos_processed' in summary: print(f"Total videos processed: {summary['total_videos_processed']}\n")
        if 'time_stats' in summary:
            print("--- End-to-End Time (seconds per video) ---")
            print(f"  Average:              {summary['time_stats']['avg_s']:.4f}")
            print(f"  Median:               {summary['time_stats']['median_s']:.4f}")
            print(f"  Standard Deviation:   {summary['time_stats']['std_dev_s']:.4f}\n")
        if 'memory_stats_gb' in summary:
            print("--- Peak GPU Memory (GB) ---")
            print(f"  Maximum:              {summary['memory_stats_gb']['max']:.4f}")
            print(f"  95th Percentile (P95):{summary['memory_stats_gb']['p95']:.4f}")
            print(f"  Median:               {summary['memory_stats_gb']['median']:.4f}\n")
        
        # <--- 修改5: 打印TTFT统计结果 ---
        if 'ttft_stats_s' in summary:
            print("--- Time to First Token (TTFT, seconds per Q&A) ---")
            print(f"  Total Q&A Events:       {summary['ttft_stats_s']['total_qa_events']}")
            print(f"  Average (Mean):         {summary['ttft_stats_s']['mean']:.4f}")
            print(f"  Standard Deviation:     {summary['ttft_stats_s']['std_dev']:.4f}")
            print(f"  Median:                 {summary['ttft_stats_s']['median']:.4f}")
            print(f"  95th Percentile (P95):    {summary['ttft_stats_s']['p95']:.4f}\n")

        if 'interactive_fps_stats' in summary and summary.get('interactive_fps_stats'):
            print("--- Interactive Encoding FPS (per QA event) ---")
            print(f"  Overall Average:      {summary['interactive_fps_stats']['overall_avg_fps']:.4f}")
            print(f"  Median of Averages:   {summary['interactive_fps_stats']['median_of_avg_fps']:.4f}\n")
        if 'decoding_throughput_stats' in summary:
            print("--- Decoding Throughput (tokens/second) ---")
            print(f"  Average (Mean):         {summary['decoding_throughput_stats']['mean_tokens_per_sec']:.2f}")
            print(f"  Standard Deviation:     {summary['decoding_throughput_stats']['std_dev_tokens_per_sec']:.2f}")
            print(f"  Median:                 {summary['decoding_throughput_stats']['median_tokens_per_sec']:.2f}\n")
        print("="*60 + "\n")
        
        # ... 保存文件的代码保持不变 ...
        base_filename = f"run_{self.chunk_idx}_of_{self.num_chunks}.jsonl"
        fixed_save_dir = '/home/cyl476530/ReKV/rebuttal'
        os.makedirs(fixed_save_dir, exist_ok=True)
        summary_output_file = os.path.join(fixed_save_dir, base_filename.replace('.jsonl', '_perf_summary.json'))
        with open(summary_output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Performance summary saved to: {summary_output_file}")


    # def load_video(self, video_path):
    #     # ---------- 1. 正常打开 ----------
    #     try:
    #         vr = VideoReader(video_path, ctx=cpu(0))
    #     except Exception as e:
    #         logger.warning(f"[VideoReader] 初始化失败 {video_path}: {e}")
    #         return None

    #     # ---------- 2. 计算采样索引 ----------
    #     avg_fps = vr.get_avg_fps()
    #     if avg_fps == 0:
    #         logger.warning(f"[VideoReader] FPS 为 0，跳过 {video_path}")
    #         return None
    #     stride = max(1, round(avg_fps / self.sample_fps))
    #     frame_idx = list(range(0, len(vr), stride))

    #     # ---------- 3. 尝试批量取帧 ----------
    #     try:
    #         video = vr.get_batch(frame_idx).asnumpy()
    #         logger.debug(f"{video_path} | shape={video.shape}")
    #         return video
    #     except Exception as e:
    #         logger.warning(f"[VideoReader] get_batch 失败 {video_path}: {e}")

    #     # ---------- 4. fault-tolerant 再试 ----------
    #     try:
    #         vr = VideoReader(video_path, ctx=cpu(0), fault_tolerant=True)
    #         video = vr.get_batch(frame_idx).asnumpy()
    #         logger.debug(f"{video_path} | shape={video.shape} (fault-tolerant)")
    #         return video
    #     except Exception as e:
    #         logger.warning(f"[VideoReader] fault_tolerant 仍失败 {video_path}: {e}")

    #     # ---------- 5. 逐帧 try / except ----------
    #     frames = []
    #     try:
    #         for idx in frame_idx:
    #             try:
    #                 frames.append(vr[idx].asnumpy())
    #             except Exception as fe:
    #                 logger.debug(f"[VideoReader] 跳过坏帧 idx={idx} | {fe}")
    #         if frames:
    #             video = np.stack(frames, axis=0)
    #             logger.debug(f"{video_path} | shape={video.shape} (逐帧 fallback)")
    #             return video
    #     except Exception as e:
    #         logger.error(f"[VideoReader] 逐帧读取彻底失败 {video_path}: {e}")

    #     # ---------- 6. 仍失败则返回 None ----------
    #     return None

    
    def calc_recall_precision(self, gt_temporal_windows, retrieved_mask): #计算召回率、精确度及 F1 值
        total_intersection_length = 0.0
    
        for (start_sec, end_sec) in gt_temporal_windows:
            start = math.floor(start_sec)
            end = math.ceil(end_sec)
            for i in range(start, end):
                if i < len(retrieved_mask) and retrieved_mask[i]:
                    intersection_start = max(start_sec, i)
                    intersection_end = min(end_sec, i + 1)
                    total_intersection_length += intersection_end - intersection_start

        gt_len = sum([end_sec - start_sec for start_sec, end_sec in gt_temporal_windows])
        retrieved_len = sum(retrieved_mask).item()

        recall = total_intersection_length / gt_len if gt_len > 0 else 0
        precision = total_intersection_length / retrieved_len if retrieved_len > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return recall, precision, f1
    
    def format_mcqa_prompt(self, question, candidates):
        assert len(question) > 0, f"Q: {question}"

        formatted_choices = "\n".join(["(" + self.choice_letters[i] + ") " + candidate for i, candidate in enumerate(candidates)])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."

        return {
            "question": f"{question}",
            "formatted_question": formatted_question,
            "prompt": self.qa_model.get_prompt(formatted_question, mc=True)
        }

    def extract_characters_regex(self, s): #提取字符正则表达式匹配结果
        s = s.strip()
        if ")" in s:
            index = s.index(")")
            pred = s[index - 1 : index]
            return pred
        else:
            return s[0]

    def video_open_qa(self, question, max_new_tokens=1024):
        pass

    def video_close_qa(self, question, candidates, correct_choice):
        pass

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        pass

    def analyze(self, debug=False):
        video_annos = self.anno[:1] if debug else self.anno
        for video_sample in tqdm(video_annos):
            logger.debug(f'video_id: {video_sample["video_id"]}')
            self.analyze_a_video(video_sample)

        dfs = []
        for (retrieve_size, chunk_size), dict_list in self.record.items():
            df = pd.DataFrame(dict_list)
            df['retrieve_size'] = retrieve_size
            df['chunk_size'] = chunk_size
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f'{self.save_dir}/{self.num_chunks}_{self.chunk_idx}.csv', index=False)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def work(QA_CLASS):
    logging.set_verbosity_error() # only show error

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="llava_ov_7b")
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--dataset", type=str, default=None, choices=['mlvu', 'qaego4d', 'egoschema', 'activitynet_qa', 'rvs_ego', 'rvs_movie', 'cgbench', 'streamingbench', 'streambench', 'ovbench', 'videomme','lvbench'])
    parser.add_argument("--compression_ratio", type=float, default=1)
    parser.add_argument("--compress_mode", type=str, default='base', choices=['offline', 'online', 'base'])
    parser.add_argument("--compress_temp", type=float, default=0)
    parser.add_argument("--encode_mode", type=str, default='dense', choices=['sparse', 'dense'])
    parser.add_argument("--retrieval_mode", type=str, default='base', choices=['offline', 'online', 'base'])
    parser.add_argument("--retrieve_temp", type=float, default=0)
    parser.add_argument('--retrieve_local', type=str, default='True', choices=['True', 'False'])
    parser.add_argument("--retrieve_local_size", type=int, default=8)
    parser.add_argument("--segment_mode", type=str, default='uniform', choices=['uniform', 'semantic'])
    parser.add_argument("--max_chunk_size", type=int, default=64)
    parser.add_argument("--min_chunk_size", type=int, default=8)
    parser.add_argument("--segment_theta", type=float, default=0.95)
    parser.add_argument("--chunk_global_sign", type=str, default='True', choices=['True', 'False'])
    args = parser.parse_args()

    if not args.debug:
        logzero.loglevel(logging.INFO)
        warnings.filterwarnings('ignore')

    os.makedirs(args.save_dir, exist_ok=True)

    # fix random seed
    random.seed(2024)
    logger.info('seed: 2024')

    # VideoQA model
    model_path = MODELS[args.model]['model_path']
    load_func = MODELS[args.model]['load_func']
    logger.info(f"Loading VideoQA model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        n_local=args.n_local,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        dataset=args.dataset,
        compression_ratio=args.compression_ratio,
        compress_mode=args.compress_mode,
        compress_temp=args.compress_temp,
        encode_mode=args.encode_mode,
        retrieval_mode=args.retrieval_mode,
        retrieve_temp=args.retrieve_temp,
        retrieve_local=args.retrieve_local,
        retrieve_local_size=args.retrieve_local_size,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        segment_theta=args.segment_theta,
        chunk_global_sign=args.chunk_global_sign,
        segment_mode=args.segment_mode,
    )

    # Load ground truth file
    anno = json.load(open(args.anno_path))

    retrieve_analyzer = QA_CLASS(
        anno=anno,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
        retrieve_size=args.retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        save_dir=args.save_dir,
    )

    retrieve_analyzer.analyze(debug=args.debug)

    retrieve_analyzer.summarize_performance()