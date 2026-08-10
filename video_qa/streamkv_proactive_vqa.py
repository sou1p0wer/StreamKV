import torch
from logzero import logger

from video_qa.streamkv_stream_vqa import StreamKVStreamVQA
from video_qa.base import work


# Verbatim from StreamingBench/src/benchmark/StreamingBenchProactive.py.
# Trigger and content queries are both wrapped in this template, exactly as the
# official protocol does.
PROMPT_TEMPLATE_PROACTIVE = '''You are an advanced image question-answering AI assistant. You have been provided with images and a question related to the images. Your task is to carefully analyze the images and provide the answer to the question. You need to carefully confirm whether the images content meet the conditions of the question, and then output the correct content.

Question: {}

The answer is:
'''


class StreamKVProactiveVQA(StreamKVStreamVQA):
    """Proactive-Output evaluation for StreamingBench.

    Proactive tasks ask the model to *emit a value at the correct moment*
    (e.g. "When the scoreboard shows 3 points for USA, output \"3\"."). StreamKV
    has no notion of "react at the right time", so we follow StreamingBench's
    official protocol: poll every frame in the [start, end] window, re-encode the
    growing clip up to each frame, ask a yes/no trigger question, and on "yes"
    ask the content question and record the trigger time.
    """

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        video_path = video_sample['video_path']
        video = self.load_video(video_path)
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            gt_output = sample['ground_truth_output']
            gt_time = float(sample['ground_truth_time'])
            start_time = float(sample['start_time'])
            end_time = float(sample['end_time'])

            start_frame = int(start_time * self.sample_fps)
            end_frame = int(end_time * self.sample_fps)

            answered_time = None
            final_answer = ''
            triggered = False
            n_polls = 0

            # Poll every frame in [start, end]; re-encode the growing clip
            # [start_frame, f] fresh at each step (Level-1, official-faithful).
            for f in range(start_frame, end_frame + 1):
                n_polls += 1
                self.qa_model.clear_cache()
                self.qa_model.encode_init_prompt()
                self.qa_model.encode_video(video_tensor[start_frame:f + 1])

                trigger_q = f'{question} Is it the right time to output "{gt_output}"? You can only answer yes or no.'
                q_text = PROMPT_TEMPLATE_PROACTIVE.format(trigger_q)
                input_text = {
                    "question": q_text,
                    "prompt": self.qa_model.get_prompt(q_text),
                }
                resp = self.qa_model.question_answering(input_text, max_new_tokens=8)

                if 'yes' in resp.strip().lower():
                    # Fresh inference for the content question (official reruns
                    # the model on the clip, stateless) — gives StreamKV its
                    # normal retrieval pass over the clip.
                    self.qa_model.clear_cache()
                    self.qa_model.encode_init_prompt()
                    self.qa_model.encode_video(video_tensor[start_frame:f + 1])
                    content_text = PROMPT_TEMPLATE_PROACTIVE.format(question)
                    content_input = {
                        "question": content_text,
                        "prompt": self.qa_model.get_prompt(content_text),
                    }
                    final_answer = self.qa_model.question_answering(content_input, max_new_tokens=32).replace('\n', '')
                    answered_time = f / self.sample_fps
                    triggered = True
                    break

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': question,
                'gt_output': gt_output,
                'gt_time': gt_time,
                'start_time': start_time,
                'end_time': end_time,
                'triggered': triggered,
                'answered_time': answered_time,
                'final_answer': final_answer,
                'n_polls': n_polls,
            })

            if 'task_type' in sample:
                self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['task_type']


if __name__ == "__main__":
    work(StreamKVProactiveVQA)
