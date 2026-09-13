"""Model profile registry. Register new family modules here to extend the dropdown."""
from . import lmstudio_model_mistral, lmstudio_model_qwen

PROFILES = {profile.NAME: profile for profile in (lmstudio_model_qwen, lmstudio_model_mistral)}


def get_profile(family):
    try:
        return PROFILES[family]
    except (KeyError, TypeError):
        raise ValueError(f"Unsupported model family. Choose one of: {', '.join(PROFILES)}.") from None


def profile_for_model(model):
    return next((profile for profile in PROFILES.values() if profile.matches(model)), lmstudio_model_qwen)


def select_family_model(client, family):
    profile = get_profile(family)
    models = client.models.list().data
    for model in models:
        if profile.matches(model.id):
            return model.id
    raise RuntimeError(
        f"LM Studio exposes no matching {family} model. Load a {family} model and ensure "
        f"its identifier contains '{family.lower()}', or change model_family."
    )
