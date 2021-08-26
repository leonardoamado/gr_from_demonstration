# from env_manager import EnvManager
from env_manager import EnvManager
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
matplotlib.use('agg')

OUT_IMG_FORMAT = 'jpg'


class TraceExtractor:
    """
    Base class used to extract traces from PDDLGym environments.
    """
    def __init__(self, env, *args, **kwargs):
        pass

    def extract(self):
        pass


class VisualTraceExtractor(TraceExtractor):
    """
    Trace extraction class that extracts visual traces
    from environments containing a renderer function.
    To see environments that have a renderer function,
    go to pddlgym/__init__.py
    """
    def __init__(self, env: EnvManager, *args, **kwargs):
        self.env = env
        _, info = self.env.reset()
        # domain_file contains the complete path to the domain
        self.domain_name = info['domain_file'].split('/')[-1].split('.')[0]
        pass

    def extract(self):
        """
        Go through all problems of a domain,
        plan the optimal path using a planner,
        and save the visual trace.
        """
        for problem_idx in range(len(self.env.problems())):
            _, info = self.env.reset(problem_idx)
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