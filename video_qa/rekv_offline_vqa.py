import torch
from logzero import logger

from video_qa.base import BaseVQA, work


class ReKVOfflineVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None): #开放性qa
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

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
            
    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video = self.load_video(video_path)
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt() # 编码初始prompt 并没有使self.init_exc=True
        self.qa_model.encode_video(video_tensor)

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            
            # QA
            if 'choices' in sample:  # CloseQA
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
            else:  # OpenQA
                qa_results = self.video_open_qa(question)
                self.record[(self.retrieve_size, self.chunk_size)].append({
                    'video_id': video_sample['video_id'],
                    'question': question,
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })

            if 'question_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']

    # def analyze_a_video(self, video_sample):
    #     """
    #     解析单个视频 + 多轮 QA。
    #     video_sample 结构:
    #         {
    #             "video_id": "xxx",
    #             "video_path": "...mp4",
    #             "conversations": [...]
    #         }
    #     """
    #     video_path = video_sample["video_path"]

    #     # 1. 读取并抽帧
    #     video = self.load_video(video_path)
    #     if video is None:
    #         logger.warning(f"[Skip] {video_path} 无法解析，直接跳过。")
    #         return

    #     # 2. numpy -> torch (若尚未是 tensor)
    #     if isinstance(video, torch.Tensor):
    #         video_tensor = video
    #     else:
    #         video_tensor = torch.from_numpy(video)

    #     # 3. 编码视频
    #     try:
    #         self.qa_model.clear_cache()
    #         self.qa_model.encode_init_prompt()          # 初始 prompt
    #         self.qa_model.encode_video(video_tensor)    # 编码视频向量
    #     except Exception as e:
    #         logger.error(f"[encode_video] 失败: {video_path} | {e}")
    #         return

    #     # 4. 遍历 QA 对话
    #     for sample in video_sample["conversations"]:
    #         logger.debug(f"sample: {sample}")

    #         question = sample["question"]
    #         answer   = sample.get("answer", None)       # 个别数据集可能没有 GT

    #         if "choices" in sample:                     # ---------- CloseQA ----------
    #             choices = sample["choices"]
    #             if answer is None:
    #                 answer = choices[0]                 # ugly fix

    #             # 计算正确选项的字母，如 ['A','B','C','D'][idx]
    #             idx_answer = choices.index(answer)
    #             if idx_answer >= len(self.choice_letters):
    #                 logger.error(f"choices 超出映射长度: {choices}")
    #                 correct_choice = None
    #             else:
    #                 correct_choice = self.choice_letters[idx_answer]

    #             qa_res = self.video_close_qa(question, choices, correct_choice)

    #             self.record[(self.retrieve_size, self.chunk_size)].append({
    #                 "video_id"      : video_sample["video_id"],
    #                 "question"      : question,
    #                 "choices"       : choices,
    #                 "answer"        : answer,
    #                 "correct_choice": correct_choice,
    #                 "pred_answer"   : qa_res["pred_answer"],
    #                 "pred_choice"   : qa_res["pred_choice"],
    #                 "qa_acc"        : qa_res["acc"] * 100,
    #                 "task"          : sample.get("question_type", None)
    #             })

    #         else:                                       # ---------- OpenQA ----------
    #             qa_res = self.video_open_qa(question)

    #             self.record[(self.retrieve_size, self.chunk_size)].append({
    #                 "video_id"    : video_sample["video_id"],
    #                 "question"    : question,
    #                 "answer"      : answer,
    #                 "pred_answer" : qa_res["pred_answer"],
    #                 "task"        : sample.get("question_type", None)
    #             })


if __name__ == "__main__":
    work(ReKVOfflineVQA)
