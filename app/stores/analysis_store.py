import os.path
import pickle
from pathlib import Path

from ..analysis.AnalysisResult import AnalysisResult
from app.config.path_config import get_analysis_data_path


def save(analysis: AnalysisResult, session_id: str):
    analysis_data_path = get_analysis_data_path(session_id)
    # create analysis path if not exists
    assert_analysis_path_exists(analysis_data_path)
    # pickle the analysis result
    path = os.path.join(analysis_data_path, analysis.filename_with_hash + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(analysis, f)


def load(filename_with_hash: str, session_id: str) -> AnalysisResult:
    analysis_data_path = get_analysis_data_path(session_id)
    path = os.path.join(analysis_data_path, filename_with_hash + '.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def assert_analysis_path_exists(analysis_data_path: str):
    analysis_path = Path(os.path.join(analysis_data_path))
    analysis_path.mkdir(parents=True, exist_ok=True)
