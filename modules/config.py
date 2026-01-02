"""
配置模块 - 全局变量和模型下载
"""

import os

MODELSCOPE_MODEL_PATH = None


def download_model_from_modelscope(model_id="damo/nlp_structbert_backbone_base_std"):
    """
    使用ModelScope下载模型到本地

    Args:
        model_id: 模型ID，默认使用阿里巴巴的StructBERT模型

    Returns:
        model_dir: 下载的模型本地路径
    """
    try:
        import modelscope
        from modelscope import snapshot_download

        cache_dir = os.path.expanduser('~/.cache/modelscope')
        os.makedirs(cache_dir, exist_ok=True)

        print(f"正在从ModelScope下载模型: {model_id}")

        model_dir = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision='master'
        )

        print(f"模型下载成功，路径: {model_dir}")
        return model_dir

    except Exception as e:
        raise RuntimeError(f"ModelScope下载失败: {e}")


def get_model_path():
    """
    获取模型路径，如果未下载则自动下载

    Returns:
        模型路径
    """
    global MODELSCOPE_MODEL_PATH
    if MODELSCOPE_MODEL_PATH is None:
        MODELSCOPE_MODEL_PATH = download_model_from_modelscope()
    return MODELSCOPE_MODEL_PATH


def set_model_path(path):
    """
    设置模型路径

    Args:
        path: 模型路径
    """
    global MODELSCOPE_MODEL_PATH
    MODELSCOPE_MODEL_PATH = path
