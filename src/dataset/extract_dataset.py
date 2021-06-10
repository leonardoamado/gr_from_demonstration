import tarfile
import pddlgym
import os
import glob
import shutil

try:
    REPO_ROOT = os.environ['RLROOT']
except KeyError as e:
    raise KeyError("Environment varible RLROOT not found. Did you run setup.sh?") from e

DATASET_ROOT = f'{REPO_ROOT}/goal-plan-recognition-dataset'

# name_map

# will only consider blocksworld for now

def add_to_pddlgym(envname, ignore_noisy=True):

    pddlgym_path = os.path.join(pddlgym.__file__, 'pddl', envname)


def extract(envname, ignore_noisy=True, obs=['100']):
    # assumes that the dataset is located at root/goal-plan-recognition-dataset
    # and only considering 100% observality

    dataset_path = os.path.join(DATASET_ROOT, envname)
    envs = find_similar(envname)
    if ignore_noisy:
        envs = list(filter(lambda x: 'noisy' not in x, envs))
    tar_files = []
    for o in obs:
        for env in envs:
            path = os.path.join(env, o)
            tars = glob.glob(path+'/*.tar.bz2')
            problems = set()
            for tar in tars:
                problem_name = tar.split('/')[-1].split('_')[0]
                problems.add(problem_name)
            for problem_name in problems:
                problem_path = os.path.join(path, problem_name)
                if os.path.exists(problem_path):
                    shutil.rmtree(problem_path)

            for fname in glob.glob(path+'/*.tar.bz2'):
                set()
                with tarfile.open(fname) as f:
                    # print(fname)
                    _extract(f)

def _extract(tar: tarfile.TarFile, hyp_idx: str, extract_all: bool):
    path = tar.name.split('/')
    name = path[-1].split('.')[0]
    dir_path = os.path.join('/'.join(path[:-1]), name)
    if extract_all and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        tar.extractall(dir_path)
    else:
        tar.extract('obs.dat', dir_path+f'/obs_hyp{hyp_idx}')

def _add_hypothesis(template_file, hyp_file):
    # change <HYPOTHESIS> for the content of real_hyp.dat
    pass

def find_similar(envname):
    return glob.glob(f'{DATASET_ROOT}/{envname}*')

# print(find_similar('blocks'))
print(extract('blocks'))