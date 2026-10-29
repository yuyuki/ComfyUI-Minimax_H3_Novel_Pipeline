"""Deterministic stable physical facts from the editable character document."""
from copy import deepcopy

from . import character_profiles as source, character_visual_selection as selection
from . import lmstudio_pipeline, util, visual_designs
from .editable_schemas import DESIGN_SCHEMA, export_schema, validate_document

VERSION = 4
PHYSICAL = ("age_appearance", "skin", "facial_features", "hair", "eyes", "build")


def prepare_profiles(data):
    profiles = selection.project(data)
    for profile in profiles:
        conflicts = {rows[0]['trait'] for rows in profile['linked_trait_conflicts']}
        evolving = {row['trait'] for row in profile['chronological_changes']}
        profile['ignored_traits'] = sorted(set(profile['ignored_traits']) | conflicts)
        profile['blocked_fields'] = sorted(set(profile['blocked_fields']) |
                                          {selection.field_for(t) for t in conflicts | evolving})
        profile['effective_traits'] = [r for r in profile['effective_traits']
            if r['trait'] not in conflicts | evolving | {'clothing', 'accessory', 'injury', 'context'}]
    return profiles


def select_profiles(data, paths):
    from .character_chapter_enrichment import source_key
    keys = {source_key(path) for path in paths}
    chapters = data['source_document']['chapters']
    for key in keys:
        if sum(source_key(c['source_file']) == key for c in chapters) != 1:
            raise ValueError('Each selected source file must identify exactly one evidence chapter.')
    if not keys:
        raise ValueError('Select at least one chapter.')
    return [p for p in prepare_profiles(data) if any(
        source_key(r['observation']['provenance']['file']) in keys for r in p['supporting_evidence'])]


def source_facts(profile):
    return {'stable_visual_description': '; '.join(dict.fromkeys(r['value'] for r in profile['effective_traits'])),
            'distinguishing_features': []}


def generate(data, paths, output, previous=None):
    profiles = select_profiles(data, paths)
    if previous is not None:
        validate_document(previous, DESIGN_SCHEMA, 'visual_designs.json')
    existing = {r['character_id']: r for r in previous['entities'] if 'character_id' in r} if previous else {}
    entities = []
    for profile in profiles:
        lmstudio_pipeline.comfy_interrupt_check()
        # Edited values in the validated profile are authoritative, never quotations
        # containing additional ignored or temporal observations.
        old = existing.get(profile['local_id'], {})
        occupied = {selection.field_for(r['trait']) for r in profile['effective_traits']}
        details = {k: v for k, v in old.get('added_details', {}).items()
                   if k in PHYSICAL and k not in set(profile['blocked_fields']) | occupied}
        entities.append({'global_id': profile['local_id'], 'character_id': profile['local_id'], 'entity_type': 'character',
            'canonical_name': profile['canonical_name'], 'aliases': profile['aliases'],
            'identity_status': profile['identity_status'],
            'source_facts': source_facts(profile),
            'added_details': details})
    designs = {'$schema': './visual_designs.schema.json',
        'schema_version': visual_designs.DESIGN_SCHEMA_VERSION,
        'image_style': previous['image_style'] if previous else 'cinematic photographic',
        'character_profiles': deepcopy(data), 'entities': entities}
    validate_document(designs, DESIGN_SCHEMA, 'visual_designs.json')
    export_schema(output, 'visual_designs')
    export_schema(output, 'character_profiles')
    util.save_json(output / 'visual_designs.json', designs)
    result = {'format_version': VERSION, 'source_digest': source._digest(data),
        'source_document': deepcopy(data), 'profiles': profiles, 'visual_designs': designs}
    saved = output / 'character_visual_profiles.json'
    util.save_json(saved, result)
    return result, str(saved)
