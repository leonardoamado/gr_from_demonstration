import tarfile
import pddlgym
import os
import glob
import shutil
import re

from adapt_blocksworld import create_problem_files, adapt_obs

try:
    REPO_ROOT = os.environ['RLROOT']
except KeyError as e:
    raise KeyError("Environment varible RLROOT not found. Did you run setup.sh?") from e

DATASET_ROOT = f'{REPO_ROOT}/goal-plan-recognition-dataset'


# will only consider blocksworld for now
def extract(envname, ignore_noisy=True, obs=['100']):
    # assumes that the dataset is located at root/goal-plan-recognition-dataset
    # and only considering 100% observality

    envs = find_similar(envname)
    if ignore_noisy:
        envs = list(filter(lambda x: 'noisy' not in x, envs))
    for o in obs:
        for env in envs:
            path = os.path.join(env, o)
            tars = glob.glob(path+'/*.tar.bz2')
            problems = set()
            for tar in tars:
                if os.path.exists(tar.split('.')[0]):
                    shutil.rmtree(tar.split('.')[0])
                problem_name = re.findall('(block-words(-\w+)?_p\d+)', tar)[0][0]
                print(problem_name)
                problems.add(problem_name)

            for problem_name in problems:
                problem_path = os.path.join(path, problem_name)
                print(problem_path)
                if os.path.exists(problem_path):
                    shutil.rmtree(problem_path)
                os.mkdir(problem_path)

            problem_map = {}
            for fname in glob.glob(path+'/*.tar.bz2'):
                problem_name = re.findall('(block-words(-\w+)?_p\d+)', fname)[0][0]
                if problem_name not in problem_map:
                    problem_map[problem_name] = ''
                    ex_all = True
                else:
                    ex_all = False
                hyp = re.findall(r'hyp-\d+', fname)[0].split('-')[-1]
                with tarfile.open(fname) as f:                    
                    _extract(f, hyp, ex_all)
            for problem_name in problems:
                problem_path = os.path.join(path, problem_name)
                pddl_paths = create_problem_files(problem_path+f'/template.pddl', problem_path+'/hyps.dat', problem_path)
                for pddl in pddl_paths:
                    pddl_name = pddl.split('/')[-1]
                    save_path = '/'.join(pddlgym.__file__.split('/')[:-1])+f'/pddl/blocks/{pddl_name}'
                    shutil.copyfile(pddl, save_path)

def _extract(tar: tarfile.TarFile, hyp_idx: str, extract_all: bool):
    path = tar.name.split('/')
    problem_name = re.findall('(block-words(-\w+)?_p\d+)', tar.name)[0][0]
    dir_path = os.path.join('/'.join(path[:-1]), problem_name)
    if extract_all:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        tar.extractall(dir_path)
    else:
        tar.extract('./obs.dat', dir_path)
    os.rename(dir_path+'/obs.dat', dir_path+f'/obs_hyp-{hyp_idx}.dat')
    adapt_obs(dir_path+f'/obs_hyp-{hyp_idx}.dat')

def find_similar(envname):
    return glob.glob(f'{DATASET_ROOT}/{envname}*')


print(extract('blocks'))