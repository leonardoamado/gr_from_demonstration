import tarfile
import pddlgym
import os
import glob

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
            for fname in glob.glob(path+'/*'):
                with tarfile.open(fname) as f:
                    _extract(f)

def _extract(tar):
    for f in tar.next():
        if  f.name
    pass

def _add_hypothesis(template_file, hyp_file):
    # change <HYPOTHESIS> for the content of real_hyp.dat
    pass

def find_similar(envname):
    return glob.glob(f'{DATASET_ROOT}/{envname}*')

# print(find_similar('blocks'))
print(extract('blocks'))