from recognizer import Recognizer

#Define datasets here
#Each element of this list is a goal recognition problem with 5 different observabilities
BLOCKS = ['output/blocks_gr/', 'output/blocks_gr2/', 'output/blocks_gr3/', 'output/blocks_gr4/']
OBS=[0.1,0.3,0.5,0.7,1.0]


def run_experiments():
    recog = Recognizer()
    blocks_results = dict()
    for obs in OBS:
        blocks_results[str(obs)] = []
    blocks_results['full'] = []
    for folder in BLOCKS:
        '''
        results is a list of tuples r_OBS
        r_n = [boolean, goal, rankings]
        '''
        results = recog.complete_recognition_folder(folder,OBS)
        #results = [[True, 1, 100],[False, 1, 100],[True, 1, 100],[True, 1, 100],[True, 1, 100]]
        for r,obs in zip(results,OBS):
            blocks_results[str(obs)].append(int(r[0]))
            blocks_results['full'].append(int(r[0]))
    print('Blocks results')
    #Print results
    for obs in OBS:
        avg = sum(blocks_results[str(obs)]) / len(blocks_results[str(obs)])
        print('OBS:', obs, 'Accuracy:', avg)
    avg_full = sum(blocks_results['full']) / len(blocks_results['full'])
    print('Average accuracy: ', avg_full)



if __name__ == "__main__":
    run_experiments()