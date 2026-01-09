import torch
import numpy as np
from logzero import logger
from decord import VideoReader, cpu
import os
from video_qa.base import BaseVQA, work
from PIL import Image
import time


class ReKVStreamVQA(BaseVQA):
    # def load_video(self, video_path):   #从指定路径加载视频文件，并根据采样帧率 (sample_fps) 对视频进行采样。
    #     if video_path.endswith('.npy'):  # FPS=1
    #         video = np.load(video_path)
    #         assert self.sample_fps <= 1
    #         num_frames = len(video)
    #         frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
    #         video = video[frame_idx]
    #     else:
    #         vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    #         fps = round(vr.get_avg_fps())
    #         frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
    #         video = vr.get_batch(frame_idx).asnumpy()
    #     return video



    def load_video(self, video_path):
        if os.path.isdir(video_path):  # 支持图片帧目录
            img_files = sorted(
                [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            )
            if len(img_files) == 0:
                raise FileNotFoundError(f"No jpg/png images found in {video_path}")
            imgs = []
            for filename in img_files:
                img = Image.open(os.path.join(video_path, filename)).convert('RGB')
                imgs.append(np.array(img))
            video = np.stack(imgs, axis=0)  # (num_frames, H, W, C)
            # 需要外部确保 self.sample_fps、帧数等一致性
            # 如需采样freq（比如每秒几帧），可在此做采样
            if  self.sample_fps < 8:
                # 按采样频率下采样
                frame_idx = np.arange(0, len(imgs), int(8 // self.sample_fps))
                video = video[frame_idx]
        elif video_path.endswith('.npy'):  # 原有支持
            video = np.load(video_path)
            assert self.sample_fps <= 1
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:  # 支持标准视频
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        return video

    def video_open_qa(self, question, max_new_tokens=1024):   #接收一个问题并生成相应的答案。
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None): #封闭性qa
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    # @torch.inference_mode()   #执行过程中禁用梯度追踪，适用于推理阶段。
    # def analyze_a_video(self, video_sample):
    #     video_path = video_sample['video_path']
    #     video_start_idx = video_end_idx = 0
    #     video = self.load_video(video_path)  #加载视频内容
    #     video_tensor = torch.from_numpy(video)

    #     self.qa_model.clear_cache()
    #     self.qa_model.encode_init_prompt()

    #     for sample in video_sample['conversations']:
    #         logger.debug(f'sample: {sample}')
    #         question = sample['question']
    #         answer = sample['answer']

    #         temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
    #         temporal_windows = temporal_windows.tolist()

    #         # encode video until receiving QA 得改 
    #         # 他这已经自动分割segment了，本身一个视频只有最后一个chunk可能会小于min_chunk_size,但是这里会将每个segment视为一个视频，导致一个视频可能有若干个小于min_chunk_size的chunk的chunk
    #         if temporal_windows[-1] > video_end_idx:
    #             video_end_idx = temporal_windows[-1]
    #             self.qa_model.encode_video(video_tensor[int(video_start_idx):int(video_end_idx)])
    #             video_start_idx = video_end_idx
        
    #         # OpenQA
    #         qa_results = self.video_open_qa(question, max_new_tokens=256)  #生成答案
    #         self.record[(self.retrieve_size, self.chunk_size)].append({
    #             'video_id': video_sample['video_id'],
    #             'question': question,
    #             'answer': answer,
    #             'pred_answer': qa_results['pred_answer'],
    #         })

    @torch.inference_mode()   #执行过程中禁用梯度追踪，适用于推理阶段。
    def analyze_a_video(self, video_sample):
        e2e_start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        per_qa_encoding_times = []
        per_qa_encoding_fps = []

        video_path = video_sample['video_path']
        video_start_idx = video_end_idx = 0
        video = self.load_video(video_path)  #加载视频内容
        video_tensor = torch.from_numpy(video)

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            
            temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
            temporal_windows = temporal_windows.tolist()
            video_end_idx = temporal_windows[-1]
            num_frames_this_qa = int(video_end_idx) - int(video_start_idx) + 1
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()

            encoding_start_time = time.perf_counter()

            self.qa_model.encode_video(video_tensor[int(video_start_idx):int(video_end_idx)])

            encoding_time_this_qa = time.perf_counter() - encoding_start_time
            fps_this_qa = num_frames_this_qa / encoding_time_this_qa if encoding_time_this_qa > 0 else 0
            per_qa_encoding_times.append(encoding_time_this_qa)
            per_qa_encoding_fps.append(fps_this_qa)
            # CloseQA
            if 'choices' in sample:
                choices = sample['choices']
                if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                    answer = choices[0]
                correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                })
            
            # OpenQA
            else:
                qa_results = self.video_open_qa(question, max_new_tokens=256)  #生成答案
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']

        e2e_total_time_s = time.perf_counter() - e2e_start_time
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        avg_encoding_time_per_qa = np.mean(per_qa_encoding_times) if per_qa_encoding_times else 0
        avg_encoding_fps_per_qa = np.mean(per_qa_encoding_fps) if per_qa_encoding_fps else 0

        print(f"e2etime:{e2e_total_time_s:.2f}s")
        print(f"峰值GPU显存占用: {peak_mem_gb:.2f} GB")
        performance_metrics = {
            'metrics_video_id': video_sample['video_id'],
            'metrics_e2e_total_time_s': e2e_total_time_s,
            'metrics_peak_memory_gb': peak_mem_gb,
            'metrics_avg_encoding_time_per_qa_s': avg_encoding_time_per_qa,
            'metrics_avg_encoding_fps_per_qa': avg_encoding_fps_per_qa,
        }
        
        if not hasattr(self, 'performance_records'):
            self.performance_records = []
        self.performance_records.append(performance_metrics)

if __name__ == "__main__":
    work(ReKVStreamVQA)