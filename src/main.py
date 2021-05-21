import sys, os

# Either do this to add pddlgym_planners to your pythonpath, or move the lib to site-packages
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)

from trace_extractor import VisualTraceExtractor
# from env_manager import EnvManager
from env_manager import EnvManager
from ml import metrics



e = EnvManager("Blocks")
print(e.env.action_space.__dict__)
v = VisualTraceExtractor(e)
v.extract()