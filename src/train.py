import spinup
from spinup.utils.run_utils import ExperimentGrid

import sys

# Command line args that will go to ExperimentGrid.run, and must possess unique
# values (therefore must be treated separately).
RUN_KEYS = ['num_cpu', 'data_dir', 'datestamp']

# Command line sweetener, allowing short-form flags for common, longer flags.
SUBSTITUTIONS = {'env': 'env_name',
                 'hid': 'ac_kwargs:hidden_sizes',
                 'act': 'ac_kwargs:activation',
                 'cpu': 'num_cpu',
                 'dt': 'datestamp'}

if __name__ == "__main__":

    algo=spinup.ppo_pytorch
    args = sys.argv[2:]

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    #获取参数列表
    arg_dict = dict()
    for i, arg in enumerate(args):
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))

    #替换参数名称缩写
    for special_name, true_name in SUBSTITUTIONS.items():
        if special_name in arg_dict:
            # swap it in arg dict
            arg_dict[true_name] = arg_dict[special_name]
            del arg_dict[special_name]

    #提取运行参数，默认：num_cpu=1, data_dir=None, datestamp=False
    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val) == 1, \
                friendly_err("You can only provide one value for %s."%k)
            run_kwargs[k] = val[0]
            del arg_dict[k]

    
    eg = ExperimentGrid(name = arg_dict['exp_name'][0])
    del arg_dict['exp_name']
    for k,v in arg_dict.items():
        eg.add(k, v)
    eg.run(algo, **run_kwargs)
