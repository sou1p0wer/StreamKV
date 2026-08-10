import torch
import numpy as np
from logzero import logger
from decord import VideoReader, cpu
import os
from video_qa.base import BaseVQA, work
from PIL import Image


class StreamKVStreamVQA(BaseVQA):

    def load_video(self, video_path):
        if os.path.isdir(video_path):
            img_files = sorted(
                [f for f in os.listdir(video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            )
            if len(img_files) == 0:
                raise FileNotFoundError(f"No jpg/png images found in {video_path}")
            imgs = []
            for filename in img_files:
                img = Image.open(os.path.join(video_path, filename)).convert('RGB')
                imgs.append(np.array(img))
            video = np.stack(imgs, axis=0)
            if  self.sample_fps < 8:
                frame_idx = np.arange(0, len(imgs), int(8 // self.sample_fps))
                video = video[frame_idx]
        elif video_path.endswith('.npy'):
            video = np.load(video_path)
            assert self.sample_fps <= 1
            num_frames = len(video)
            frame_idx = np.linspace(0, num_frames-1, int(num_frames*self.sample_fps), dtype=int).tolist()
            video = video[frame_idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), int(fps / self.sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
        return video

    def video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video_path = video_sample['video_path']
        video_start_idx = video_end_idx = 0
        video = self.load_video(video_path)
        video_tensor = torch.from_numpy(video)

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            temporal_windows = torch.tensor([sample['start_time'], sample['end_time']]) * self.sample_fps
            temporal_windows = temporal_windows.tolist()
            video_end_idx = temporal_windows[-1]
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()

            self.qa_model.encode_video(video_tensor[int(video_start_idx):int(video_end_idx)])
            # CloseQA
            if 'choices' in sample:
                choices = sample['choices']
                if answer is None:
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
                qa_results = self.video_open_qa(question, max_new_tokens=256)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']

if __name__ == "__main__":
    work(StreamKVStreamVQA)
