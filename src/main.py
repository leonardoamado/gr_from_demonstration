from trace_extractor import VisualTraceExtractor
# from env_manager import EnvManager
from env_manager import EnvManager
from ml import metrics

env = EnvManager("Blocks")
print(env.env.action_space.__dict__)
v = VisualTraceExtractor(env)
v.extract()
