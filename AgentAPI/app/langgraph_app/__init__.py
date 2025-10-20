from pathlib import Path
import sys

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
	sys.path.append(str(_PARENT))
