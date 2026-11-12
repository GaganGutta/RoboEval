"""Compatibility shims for robomimic v0.5.0. Import this BEFORE any robomimic module.

1. Language stub. robomimic/utils/lang_utils.py loads a CLIP-ViT-L/14 text model from
   Hugging Face at *import time*. Every training and rollout process pays for it in memory
   (it crashed WSL with 6 parallel rollout workers) even though our low-dim policies are
   not language-conditioned. We register a stub module under the same name exposing the
   attributes robomimic uses (checked with grep for LangUtils.* in robomimic v0.5.0).
   get_lang_emb always returns None: env metadata carries a placeholder lang string
   ("dummy"), and the env only uses the embedding to add a "lang_emb" observation, which
   our policies never read. get_lang_emb_shape raises, because it is only called when a
   policy requests "lang_emb" as an input, which would be a real use of language.
"""
import sys
import types


def _install_lang_stub():
    name = "robomimic.utils.lang_utils"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.LANG_EMB_OBS_KEY = "lang_emb"
    stub.LANG_COND_ENABLED = False

    def get_lang_emb(lang):
        return None

    def get_lang_emb_shape():
        raise RuntimeError("language conditioning disabled by train/robomimic_compat.py")

    stub.get_lang_emb = get_lang_emb
    stub.get_lang_emb_shape = get_lang_emb_shape
    sys.modules[name] = stub


_install_lang_stub()
