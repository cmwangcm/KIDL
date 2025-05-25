import subprocess
from time import time
from utils import *

def run_script(command):
    printf(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    stderr = process.communicate()[1]
    if stderr:
        print("Error:", stderr.strip())

def run_get_data(data_dir, dataset):
    command = 'python llm/models/get_data.py --data_dir {data_dir} --dataset {dataset}'.format(data_dir=data_dir, dataset=dataset)
    run_script(command)

def run_dnn_sg_pipeline(model, dataset_t, llm, suffix, learning_rate=1e-3, mono_factor=0., stb_factor=1., stop_num=10, need_train=True, need_predict=True):
    command = 'python llm/models/dnn_sg/pipeline.py --model {model} --dataset_t {dataset_t} --llm {llm} --suffix {suffix} --learning_rate {learning_rate} --mono_factor {mono_factor} --stop_num {stop_num} --need_train {need_train} --need_predict {need_predict}'.format(model=model, dataset_t=dataset_t, llm=llm, suffix=suffix, learning_rate=learning_rate, mono_factor=mono_factor, stop_num=stop_num, need_train=need_train, need_predict=need_predict)
    run_script(command)


if __name__ == '__main__':
    run_get_data('dnn_sg/data/', 'ngsim_i80')
    run_dnn_sg_pipeline('dnn_sg_mode_mono_cons', 'ngsim_i80', llm='deepseek-coder', suffix='zero-shot-dist', mono_factor=5000., stb_factor=0.9, learning_rate=1e-3, stop_num=10, need_train=True, need_predict=True)