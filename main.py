import yaml
import os
import pickle
import json
import pandas as pd

from utils.public_functions import load_samples
from models.train import train
from models.evaluation import get_eval_df
import warnings
warnings.filterwarnings('ignore')

# config
dataset = 'D1'
config_file = f'{dataset}.yaml'
config = yaml.load(open(f'config/{config_file}', 'r'), Loader=yaml.FullLoader)
print('load config.')
res_dir = f'res/{dataset}'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
naive_model_path = f'{res_dir}/naive_model.pkl'
fd_model_path = f'{res_dir}/fd_model.pkl'
test_df_path = f'{res_dir}/test_df.csv'
fd_test_df_path = f'{res_dir}/fd_test_df.csv'
res_path = f'{res_dir}/res.json'
# load samples
train_samples, test_samples = load_samples(config['path']['sample_dir'])
print('load samples.')
input_samples = train_samples if config['train_samples_num'] == 'whole' else train_samples[: config['train_samples_num']]
# train naive model
print(f"train samples num: {len(input_samples)}, aug_multiple: {config['model_param']['aug_multiple']}")
model = train(input_samples, config['model_param'])
print('naive model trained.')
with open(naive_model_path, 'wb') as f:
    pickle.dump(model, f)
# train feedback model and use the model with feedback for evaluation
cases = pd.read_csv(config['path']['case_dir'])
fd_model, test_df, fd_test_df, res_dict = get_eval_df(model, cases, test_samples, config)
# save the result
with open(fd_model_path, 'wb') as f:
    pickle.dump(fd_model, f)
test_df.to_csv(test_df_path, index=False)
fd_test_df.to_csv(fd_test_df_path, index=False)
with open(res_path, 'w') as f:
    json.dump(res_dict, f)
