from recognizer import Recognizer, StateQmaxRecognizer
from ml.metrics import *
#from tqdm import tqdm

# Define datasets here
# Each element of this list is a goal recognition problem with 5 different observabilities
BLOCKS = ['output/blocks_gr/','output/blocks_gr2/','output/blocks_gr3/','output/blocks_gr4/','output/blocks_gr5/',
    'output/blocks_gr6/','output/blocks_gr7/','output/blocks_gr8/','output/blocks_gr9/','output/blocks_gr10/']

HANOI = ['output/hanoi_gr/','output/hanoi_gr2/','output/hanoi_gr3/','output/hanoi_gr4/','output/hanoi_gr5/',
    'output/hanoi_gr6/','output/hanoi_gr7/','output/hanoi_gr8/','output/hanoi_gr9/','output/hanoi_gr10/']

SKGRID = ['output/skgrid_gr/','output/skgrid_gr2/','output/skgrid_gr3/','output/skgrid_gr4/','output/skgrid_gr5/',
    'output/skgrid_gr6/','output/skgrid_gr7/','output/skgrid_gr8/','output/skgrid_gr9/','output/skgrid_gr10/']
    
OBS = [0.1, 0.3, 0.5, 0.7, 1.0]


class Experiment:
    # TODO Fill this in
    """A single experiment for one recognizer"""
    def __init__(self, recognizer: Recognizer):
        self.recognizer = recognizer
        self.problems = 0
        self.totalTime = 0

    def reset(self):
        pass

    def run_experiment(self, options: dict):
        pass

    def compute_stats(self):
        pass

def check_draw(ranking):
    return ranking[0][1] == ranking[1][1]

def check_spread(ranking):
    head = ranking[0]
    tail = ranking[1:]
    spread = 1
    for goal_value in tail:
        if goal_value[1] == head[1]:
            spread += 1
        else:
            break
    return spread


def run_experiments(train=True, even_punish=False):
    recog = Recognizer(evaluation=soft_divergence_point)
    blocks_results = dict()
    for obs in OBS:
        blocks_results[str(obs)] = []
    blocks_results['full'] = []
    for folder in BLOCKS:
        '''
        results is a list of tuples r_OBS
        r_n = [boolean, goal, rankings]
        '''
        if train:
            results = recog.complete_recognition_folder(folder, OBS)
        else:
            results = recog.only_recognition_folder(folder, OBS)
        for r, obs in zip(results, OBS):
            prediction = r[0]
            if r[0] and even_punish:
                prediction = 1/check_spread(r[-1])
                #prediction = not check_draw(r[-1])
                 

            blocks_results[str(obs)].append(float(prediction))
            blocks_results['full'].append(float(prediction))
    print('Blocks results')
    # Print results
    for obs in OBS:
        avg = sum(blocks_results[str(obs)]) / len(blocks_results[str(obs)])
        print('OBS:', obs, 'Accuracy:', avg)
    avg_full = sum(blocks_results['full']) / len(blocks_results['full'])
    print('Average accuracy: ', avg_full)


if __name__ == "__main__":
    run_experiments(False, True)
