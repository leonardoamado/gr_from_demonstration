# from env_manager import EnvManager
from env_manager import EnvManager
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
import os

OUT_IMG_FORMAT = 'jpg'

class TraceExtractor:
    def __init__(self, env, *args, **kwargs):
        pass

    def extract(self):
        pass


class VisualTraceExtractor(TraceExtractor):
    def __init__(self, env: EnvManager, *args, **kwargs):
        self.env = env
        _, info = self.env.reset()
        self.domain_name = info['domain_file'].split('/')[-1].split('.')[0]
        pass

    def extract(self):
        # self.env.
        for problem_idx in range(len(self.env.problems())):
            _, info = self.env.reset(problem_idx)
            # domain = info['domain_file']
            print('Problem file: ', info['problem_file'])
            problem_name = info['problem_file'].split('/')[-1].split('.')[0]
            trace = self.env.render()
            trace_name = f'{self.domain_name}_{problem_name}_0'
            self.save_trace(trace, self.domain_name, problem_name, trace_name)
            plan = self.env.plan()
            for i, a in enumerate(plan, start=1):
                trace_name = f'{self.domain_name}_{problem_name}_{i}'
                self.env.step(a)
                trace = self.env.render()
                self.save_trace(trace, self.domain_name, problem_name, trace_name)
                plt.close('all')


    def save_trace(self, trace, domain, problem, trace_name):
        dataset_path = os.environ['DATASETPATH']
        problem_path = f'{dataset_path}/{domain}/{problem}'
        os.makedirs(problem_path, exist_ok=True)
        cv2.imwrite(f'{dataset_path}/{domain}/{problem}/{trace_name}.{OUT_IMG_FORMAT}', trace)