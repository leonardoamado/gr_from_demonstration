from ml.rl import TabularDynaQLearner
from recognizer import Recognizer, StateQmaxRecognizer, ActionQmaxRecognizer
from ml.metrics import *
# from tqdm import tqdm
from ml.metrics import kl_divergence_norm_softmax, divergence_point, soft_divergence_point, trajectory_q_value

# Define datasets here
# Each element of this list is a goal recognition problem with 5 different observabilities
BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/', 'output/blocks_gr5/',
          'output/blocks_gr6/', 'output/blocks_gr7/', 'output/blocks_gr8/', 'output/blocks_gr9/', 'output/blocks_gr10/']

HANOI = ['output/hanoi_gr/', 'output/hanoi_gr2/', 'output/hanoi_gr3/', 'output/hanoi_gr4/', 'output/hanoi_gr5/',
         'output/hanoi_gr6/', 'output/hanoi_gr7/', 'output/hanoi_gr8/', 'output/hanoi_gr9/', 'output/hanoi_gr10/']

SKGRID = ['output/skgrid_gr/', 'output/skgrid_gr2/', 'output/skgrid_gr3/', 'output/skgrid_gr4/', 'output/skgrid_gr5/',
          'output/skgrid_gr6/', 'output/skgrid_gr7/', 'output/skgrid_gr8/', 'output/skgrid_gr9/', 'output/skgrid_gr10/']

#OBS = [0.1, 0.3, 0.5, 0.7, 1.0]

# USE THIS FOR NOISY
OBS = [0.5, 1.0]


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
                # prediction = not check_draw(r[-1])

            blocks_results[str(obs)].append(float(prediction))
            blocks_results['full'].append(float(prediction))
    print('Blocks results')
    # Print results
    for obs in OBS:
        avg = sum(blocks_results[str(obs)]) / len(blocks_results[str(obs)])
        print('OBS:', obs, 'Accuracy:', avg)
    avg_full = sum(blocks_results['full']) / len(blocks_results['full'])
    print('Average accuracy: ', avg_full)


def run_experiments_domain(recog: Recognizer, domain: List[str], train=True, even_punish=False):
    # recog = Recognizer(evaluation=soft_divergence_point)
    domain_results = dict()
    for obs in OBS:
        domain_results[str(obs)] = []
    domain_results['full'] = []
    for folder in domain:
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
                # prediction = not check_draw(r[-1])

            domain_results[str(obs)].append(float(prediction))
            domain_results['full'].append(float(prediction))
    return domain_results


def measure_confusion(r_output):
    prediction = r_output[0]
    ranking = r_output[-1]

    head = ranking[0]
    tail = ranking[1:]
    fn = int(not prediction)
    fp = 0
    tn = 0
    if prediction:       
        for goal_value in tail:
            if goal_value[1] == head[1]:
                fp += 1
            else:
                tn += 1    
    else:
        fp = 1
        for goal_value in tail[:-1]:
            if goal_value[1] == head[1]:
                fp += 1
            else:
                tn += 1

    #      tp               fn                   fp  tn       
    return int(prediction), fn, fp, tn


def run_experiments_domain_all_metrics(recog: Recognizer, domain: List[str], train=True):
    # recog = Recognizer(evaluation=soft_divergence_point)
    domain_results = dict()
    for obs in OBS:
        domain_results[str(obs)] = dict()
    domain_results['full'] = dict()
    for key in domain_results:
        domain_results[key]['TP'] = 0
        domain_results[key]['FP'] = 0
        domain_results[key]['FN'] = 0
        domain_results[key]['TN'] = 0
        domain_results[key]['len'] = 0
        # domain_results[key]['opt'] = 0
    for folder in domain:
        '''
        results is a list of tuples r_OBS
        r_n = [boolean, goal, rankings]
        '''
        if train:
            results = recog.complete_recognition_folder(folder, OBS)
        else:
            results = recog.only_recognition_folder(folder, OBS)
        for r, obs in zip(results, OBS):
            tp, fn, fp, tn = measure_confusion(r)      
            domain_results[str(obs)]['TP'] += tp
            domain_results[str(obs)]['FP'] += fp
            domain_results[str(obs)]['FN'] += fn
            domain_results[str(obs)]['TN'] += tn
            domain_results[str(obs)]['len'] += 4
            domain_results['full']['TP'] += tp
            domain_results['full']['FP'] += fp
            domain_results['full']['FN'] += fn
            domain_results['full']['TN'] += tn
            domain_results['full']['len'] += 4

    return domain_results


def calculate_all_metrics(obs_metrics):
    accuracy = 0
    precision = 0
    recall = 0
    fscore = 0
    accuracy = (obs_metrics['TP'] + obs_metrics['TN']) / obs_metrics['len']
    precision = obs_metrics['TP'] / (obs_metrics['TP'] + obs_metrics['FP'])
    recall = obs_metrics['TP'] / (obs_metrics['TP'] + obs_metrics['FN'])
    if precision + recall != 0:
        fscore = (2 * precision * recall) / (precision + recall)
    else:
        fscore = 0
    return accuracy, precision, recall, fscore


def run_all_domains_metrics(train=True, recog=Recognizer(), file=None):
    print(f"******  Running all domains for {recog} ******")
    blocks = run_experiments_domain_all_metrics(recog, BLOCKS, train)
    hanoi = run_experiments_domain_all_metrics(recog, HANOI, train)
    skgrid = run_experiments_domain_all_metrics(recog, SKGRID, train)

    if file:
        file = open(file, 'a')
        # file.write(f"******  Results for {recog} ******\n")
        file.write(f"# {recog} \n")
    print(f"******  Results for {recog} ******")

    print('Blocks results')
    #print(f'Optimality Ratio: {recog.folder_opt_ratio(BLOCKS[0])}')
    if file:
        # file.write(f"\n ** Blocks Results ** \n")
        file.write(f"\n# ** Blocks Results ** \n")
        file.write(f'#OBS\t Acc\t Prec\t Rec\t F-S\n')
    for obs in OBS:
        accuracy, precision, recall, fscore = calculate_all_metrics(blocks[str(obs)])
        print('OBS:', obs, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
        if file:
            # file.write(f'OBS: {obs} Accuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fscore}\n')
            file.write(f'{obs}\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n')
    accuracy, precision, recall, fscore = calculate_all_metrics(blocks['full'])
    print('Averages - Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
    if file:
        # file.write(f'Averages - Accuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fscore}\n')
        file.write(f'Avg\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n')

    print('Hanoi results')
    #print(f'Optimality Ratio: {recog.folder_opt_ratio(HANOI[0])}')
    if file:
        # file.write(f"\n ** Hanoi Results ** \n")
        file.write(f"\n# ** Hanoi Results ** \n")
        file.write(f'#OBS\t Acc\t Prec\t Rec\t F-S\n')
    for obs in OBS:
        accuracy, precision, recall, fscore = calculate_all_metrics(hanoi[str(obs)])
        print('OBS:', obs, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
        if file:
            # file.write(f'OBS: {obs} Accuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fscore}\n')
            file.write(f'{obs}\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n')
    accuracy, precision, recall, fscore = calculate_all_metrics(hanoi['full'])
    print('Averages - Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
    if file:
        # file.write(f'Averages - Accuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fscore}\n')
        file.write(f'Avg\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n')

    print('SkGrid results')
    #print(f'Optimality Ratio: {recog.folder_opt_ratio(SKGRID[0])}')
    if file:
        # file.write(f"\n ** SkGrid Results ** \n")
        file.write(f"\n# ** SkGrid Results ** \n")
        file.write(f'#OBS\t Acc\t Prec\t Rec\t F-S\n')
    for obs in OBS:
        accuracy, precision, recall, fscore = calculate_all_metrics(skgrid[str(obs)])
        print('OBS:', obs, 'Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
        if file:
            # file.write(f'OBS: {obs} Accuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fscore}\n')
            file.write(f'{obs}\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n')
    accuracy, precision, recall, fscore = calculate_all_metrics(skgrid['full'])
    print('Averages - Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall, 'F-Score:', fscore)
    if file:
        # file.write(f'Averages - Accuracy: {accuracy} Precision: {precision} Recall: {recall} F-Score: {fscore}\n')
        file.write(f'Avg\t{accuracy:.2f}\t{precision:.2f}\t{recall:.2f}\t{fscore:.2f}\n')


def run_all_domains(train=True, recog=Recognizer()):
    blocks = run_experiments_domain(recog, BLOCKS, train, True)
    hanoi = run_experiments_domain(recog, HANOI, train, True)
    skgrid = run_experiments_domain(recog, SKGRID, train, True)

    print('Blocks results')
    # Print results
    for obs in OBS:
        avg = sum(blocks[str(obs)]) / len(blocks[str(obs)])
        print('OBS:', obs, 'Precision:', avg)
    avg_full = sum(blocks['full']) / len(blocks['full'])
    print('Average precision: ', avg_full)

    print('Hanoi results')
    # Print results
    for obs in OBS:
        avg = sum(hanoi[str(obs)]) / len(hanoi[str(obs)])
        print('OBS:', obs, 'Precision:', avg)
    avg_full = sum(hanoi['full']) / len(hanoi['full'])
    print('Average precision: ', avg_full)

    print('SkGrid results')
    # Print results
    for obs in OBS:
        avg = sum(skgrid[str(obs)]) / len(skgrid[str(obs)])
        print('OBS:', obs, 'Precision:', avg)
    avg_full = sum(skgrid['full']) / len(skgrid['full'])
    print('Average precision: ', avg_full)


if __name__ == "__main__":
    # run_experiments(False, True)
    # run_all_domains(train=False, recog=Recognizer(evaluation=trajectory_q_value))

    # I used this to test noise
    # run_all_domains_metrics(train=False, recog=Recognizer())

    # recog = Recognizer()
    # results = recog.only_recognition_folder(BLOCKS[1], [0.5,1.0])

    for recognizer in [Recognizer(evaluation=kl_divergence_norm_softmax),
                        Recognizer(evaluation=soft_divergence_point),
                        Recognizer(evaluation=trajectory_q_value),
                        StateQmaxRecognizer(),
                        ActionQmaxRecognizer()
                       ]:
        run_all_domains_metrics(train=False, recog=recognizer, file='results_noisy.txt')
    # for recognizer in [Recognizer(evaluation=kl_divergence_norm_softmax, method=TabularDynaQLearner),
    #                    Recognizer(evaluation=soft_divergence_point, method=TabularDynaQLearner),
    #                    Recognizer(evaluation=trajectory_q_value, method=TabularDynaQLearner),
    #                    StateQmaxRecognizer(method=TabularDynaQLearner),
    #                    ActionQmaxRecognizer(method=TabularDynaQLearner),
    #                    ]:
    #     run_all_domains_metrics(train=True, recog=recognizer, file='results.txt')
