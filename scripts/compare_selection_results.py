import json
import math
import numpy as np

def load_dump(path):
    with open(path,'r') as f:
        return json.load(f)

def compute_metrics(selected, gt):
    # selected and gt are lists of numbers
    selected = np.array(selected, dtype=float)
    gt = np.array(gt, dtype=float)
    mae = np.mean(np.abs(selected - gt))
    rmse = np.sqrt(np.mean((selected - gt)**2))
    # compute masked MAPE to avoid division by zero instability
    mask = np.abs(gt) > 1e-8
    if mask.any():
        mape = np.mean(np.abs((selected[mask] - gt[mask]) / gt[mask])) * 100
    else:
        mape = float('nan')
    return mae, rmse, mape

opt = load_dump('outputs/dump/my_data_choices.json')
llm = load_dump('outputs/dump/my_data_choices_update1.json')

# Build arrays of selected values and gt for each dump

def build_selected_and_gt(dump, use_optimal=True):
    sel_vals = []
    gts = []
    for entry in dump:
        choices = entry['choices']
        gt = entry['ground_truth']
        # choices are lists (sequence), choose first element if seq length 1
        if use_optimal:
            idx = entry.get('optimal_idx', 0)
        else:
            # for llm dump we may not have final_answer; assume optimal_idx used
            idx = entry.get('optimal_idx', 0)
        # choices[idx] may be [value] or list
        val = choices[idx]
        # if sequence - take first
        if isinstance(val, list):
            val = val[0]
        if isinstance(gt, list):
            gt = gt[0]
        sel_vals.append(float(val))
        gts.append(float(gt))
    return sel_vals, gts

opt_sel, opt_gt = build_selected_and_gt(opt, use_optimal=True)
llm_sel, llm_gt = build_selected_and_gt(llm, use_optimal=True)

mae_o, rmse_o, mape_o = compute_metrics(opt_sel, opt_gt)
mae_l, rmse_l, mape_l = compute_metrics(llm_sel, llm_gt)

print('Optimal dump metrics: MAE={:.3f}, RMSE={:.3f}, MAPE={:.3f}'.format(mae_o, rmse_o, mape_o))
print('LLM dump metrics:     MAE={:.3f}, RMSE={:.3f}, MAPE={:.3f}'.format(mae_l, rmse_l, mape_l))

# Per-node breakdown
def per_node(dump):
    from collections import defaultdict
    vals = defaultdict(list)
    gts = defaultdict(list)
    for e in dump:
        node = e['node_idx']
        idx = e.get('optimal_idx', 0)
        val = e['choices'][idx]
        if isinstance(val, list): val = val[0]
        gt = e['ground_truth']
        if isinstance(gt, list): gt = gt[0]
        vals[node].append(float(val))
        gts[node].append(float(gt))
    res = {}
    for n in vals:
        res[n] = compute_metrics(vals[n], gts[n])
    return res

opt_per = per_node(opt)
llm_per = per_node(llm)
print('\nPer-node metrics (node: MAE, RMSE, MAPE):')
for n in sorted(opt_per.keys()):
    om, orr, op = opt_per[n]
    lm, lrr, lp = llm_per[n]
    print(f'node {n} optimal: MAE={om:.3f}, RMSE={orr:.3f}, MAPE={op if not math.isnan(op) else "nan"} ; llm: MAE={lm:.3f}, RMSE={lrr:.3f}, MAPE={lp if not math.isnan(lp) else "nan"}')

# sample-wise comparison: count wins
opt_better = 0
llm_better = 0
equal = 0
for i in range(min(len(opt_sel), len(llm_sel))):
    e_opt = abs(opt_sel[i] - opt_gt[i])
    e_llm = abs(llm_sel[i] - llm_gt[i])
    if e_opt < e_llm:
        opt_better += 1
    elif e_llm < e_opt:
        llm_better += 1
    else:
        equal += 1

print(f'\nSample-wise: optimal better: {opt_better}, llm better: {llm_better}, equal: {equal}')
