import sys, os

# Either do this to add pddlgym_planners to your pythonpath, or move the lib to site-packages
sys.path.append(os.path.abspath(os.path.join('.')))

from trace_extractor import VisualTraceExtractor
from env_manager import EnvManager




EnvManager("Blocks")
VisualTraceExtractor()