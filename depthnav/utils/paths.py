import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_DATASETS_ROOT = Path('/root/gpufree-data/datasets')
DEFAULT_LOCAL_DATASETS_ROOT = REPO_ROOT / 'datasets'


def _normalize(path: Path) -> str:
    return str(path.expanduser().resolve())


def get_datasets_root(require_exists: bool = True) -> str:
    candidates = []

    env_root = os.environ.get('DEPTHNAV_DATASETS_ROOT')
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(DEFAULT_SHARED_DATASETS_ROOT)
    candidates.append(DEFAULT_LOCAL_DATASETS_ROOT)

    seen = set()
    ordered = []
    for path in candidates:
        normalized = _normalize(path)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(Path(normalized))

    for path in ordered:
        if not require_exists or path.exists():
            return str(path)

    return str(ordered[0])


def get_depthnav_dataset_path(require_exists: bool = True) -> str:
    env_path = os.environ.get('DEPTHNAV_DATASET_PATH')
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path(get_datasets_root(require_exists=False)) / 'depthnav_dataset')

    seen = set()
    ordered = []
    for path in candidates:
        normalized = _normalize(path)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(Path(normalized))

    for path in ordered:
        if not require_exists or path.exists():
            return str(path)

    return str(ordered[0])


def resolve_depthnav_dataset_path(dataset_path: Optional[str], require_exists: bool = True) -> str:
    if dataset_path in (None, ''):
        return get_depthnav_dataset_path(require_exists=require_exists)
    return _normalize(Path(dataset_path))


def get_depthnav_dataset_subpath(*parts: str, require_exists: bool = True) -> str:
    return os.path.join(get_depthnav_dataset_path(require_exists=require_exists), *parts)


def get_depthnav_scene_dataset_config_path(require_exists: bool = True) -> str:
    dataset_path = Path(get_depthnav_dataset_path(require_exists=require_exists))
    matches = sorted(dataset_path.glob('*.scene_dataset_config.json'))
    if matches:
        return _normalize(matches[0])
    return _normalize(dataset_path / 'depthnav_dataset.scene_dataset_config.json')


def get_depthnav_agent_object_path(require_exists: bool = True) -> str:
    return get_depthnav_dataset_subpath(
        'configs', 'agents', 'DJI_Mavic_Mini_2.object_config.json',
        require_exists=require_exists,
    )
