import os
import argparse
import pandas as pd


# StreamingBench proactive tolerance: a trigger is "on time" if it fires within
# ±2s of the ground-truth time (official src/data/count.py:55).
TIME_TOLERANCE = 2.0


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--results_path', type=str, default=None)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.results_path is not None:
    df = pd.read_csv(args.results_path)
    args.save_dir = os.path.dirname(args.results_path)
else:
    df = pd.read_csv(os.path.join(args.save_dir, 'results.csv'))

total = len(df)
time_correct = 0
answer_correct = 0
n_triggered = 0

for _, row in df.iterrows():
    gt_time = float(row['gt_time'])
    answered_time = row['answered_time']
    final_answer = '' if pd.isna(row['final_answer']) else str(row['final_answer'])
    gt_output = str(row['gt_output'])

    triggered = pd.notna(answered_time)
    if triggered:
        n_triggered += 1

    # Replicates official count.py:55-58. Divergence: official scores time-correct
    # from the last polled time even when never triggered (its gt+4 poll bound
    # keeps that outside ±2s); our [start, end] bound can end within ±2s of gt,
    # so we count time-correct only if actually triggered — no false positives.
    if triggered and abs(float(answered_time) - gt_time) <= TIME_TOLERANCE:
        time_correct += 1
        # Answer is nested: only counts when the time was already correct.
        if gt_output in final_answer:
            answer_correct += 1

time_acc = time_correct / total if total > 0 else 0.0
answer_acc = answer_correct / total if total > 0 else 0.0

print(f'#Samples: {total}')
print(f'#Triggered: {n_triggered}')
print(f'time_accuracy:   {time_acc * 100:.2f}% ({time_correct}/{total})')
print(f'answer_accuracy: {answer_acc * 100:.2f}% ({answer_correct}/{total})')
print(f'save_dir: {args.save_dir}')

if args.debug:
    for _, row in df.iterrows():
        print(f"  video={row['video_id']} gt={row['gt_time']}s/{str(row['gt_output'])!r} "
              f"answered={row['answered_time']}s pred={str(row['final_answer'])!r} "
              f"polls={row['n_polls']}")
