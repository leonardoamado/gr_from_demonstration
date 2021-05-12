from env_manager import EnvManager
import cv2

class TraceExtractor:
    def __init__(self, env, *args, **kwargs):
        pass

    def extract(self):
        pass


class VisualTraceExtractor(TraceExtractor):
    def __init__(self, env: EnvManager, *args, **kwargs):
        self.env = env
        pass

    def extract(self):
        for problem_idx in range(len(self.env.problems())):
            obs, info = self.env.reset(problem_idx)
            obss = [obs]
            plan = self.env.plan()
            self.env.execute_plan()

    def save_trace(self, trace, trace_name):

        # cv2.imwrite()
        pass