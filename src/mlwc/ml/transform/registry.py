# モデル名(str)とTransformクラスを紐付ける辞書
TRANSFORM_REGISTRY = {}


def register_transform(name):
    """新しいTransformを台帳に登録するためのデコレータ"""

    def decorator(cls):
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator
