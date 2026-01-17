import json
import os

prev_path = os.path.join('outputs','dump','my_data_choices.json')
out_dir = os.path.join('data','llm_output')
out_path = os.path.join(out_dir,'my_data_output.json')

with open(prev_path,'r') as f:
    data = json.load(f)

out = []
for e in data:
    opt = e.get('optimal_idx')
    if opt is None:
        out.append({'final_answer': None})
    else:
        out.append({'final_answer': int(opt) + 1})

os.makedirs(out_dir, exist_ok=True)
with open(out_path,'w') as f:
    json.dump(out, f)
print('Wrote', out_path, 'entries:', len(out))
