"""Run GPT-SoVITS on loopback with the supplied CPU model configuration."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / '.runtime/GPT-SoVITS'
os.chdir(REPO)
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ['NLTK_DATA'] = str(REPO / 'runtime/nltk_data')
os.environ['PYTHONUTF8'] = '1'
sys.path.insert(0, str(REPO))
# The pure-Python implementation avoids compiling jieba_fast on Windows.
import jieba
import jieba.posseg
sys.modules['jieba_fast'] = jieba
sys.modules['jieba_fast.posseg'] = jieba.posseg
sys.argv = ['api_v2.py', '-a', '127.0.0.1', '-p', '9880', '-c', str(ROOT / 'scripts/voice-cpu.yaml')]
runpy.run_path(str(REPO / 'api_v2.py'), run_name='__main__')
