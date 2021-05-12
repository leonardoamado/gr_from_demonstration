# from env_manager import EnvManager
from .env_manager import EnvManager
import cv2
import os

class TraceExtractor:
    def __init__(self, env, *args, **kwargs):
        pass

    def extract(self):
        pass


class VisualTraceExtractor(TraceExtractor):
    def __init__(self, env: EnvManager, *args, **kwargs):
        self.env = env
        _, info = self.env.reset()
        self.domain_name = info['domain_file']
        pass

    def extract(self):
        # self.env.
        for problem_idx in range(len(self.env.problems())):
            _, info = self.env.reset(problem_idx)
            # domain = info['domain_file']
            problem_name = info['problem_file'].split('/')[-1].split('.')[0]
            # obss = [obs]
            plan = self.env.plan()
            for i, a in enumerate(plan):
                trace_name = f'{self.domain_name}_{problem_name}_{i}'
                self.env.step(a)
                trace = self.env.render()
                self.save_trace(trace, trace_name)



    def save_trace(self, trace, domain, problem, trace_name):
        dataset_path = os.environ['DATASETPATH']
        cv2.imwrite(f'{dataset_path}/{domain}/{problem}/{trace_name}', trace)