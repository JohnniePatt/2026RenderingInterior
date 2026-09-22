"""Resolve semantic search concepts without overwriting human annotations."""
from services.prompt_resolution import conflict


def resolve_concept(entity):
    explicit = entity.get('validation_concept')
    warning = 'Semantic metadata conflict — please verify Validation Concept.' if conflict(entity) else None
    if isinstance(explicit, str) and explicit.strip():
        return {'concept': explicit.strip(), 'source': 'manual', 'warning': warning, 'needs_review': False}
    if warning:
        return {'concept': '', 'source': 'conflict', 'warning': warning, 'needs_review': True}
    for field in ('category', 'semantic', 'description'):
        text = entity.get(field)
        if isinstance(text, str) and text.strip() and text.strip().lower() not in ('other', 'unknown'):
            return {'concept': text.strip().replace('_', ' '), 'source': field, 'warning': None, 'needs_review': False}
    return {'concept': '', 'source': 'missing', 'warning': 'Validation Concept is required.', 'needs_review': True}
