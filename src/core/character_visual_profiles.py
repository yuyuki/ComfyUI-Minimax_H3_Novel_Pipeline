"""Deterministic stable physical facts from the editable character document."""
from copy import deepcopy

from . import character_visual_selection as selection
from . import character_evidence, lmstudio_pipeline, util, visual_designs
from .character_profile_document import validate_document as validate_profiles
from .editable_schemas import DESIGN_SCHEMA, export_schema, validate_document

VERSION = 5
PHYSICAL = ("age_appearance", "skin", "facial_features", "hair", "eyes", "build")


def prepare_profiles(data):
    profiles = selection.project(data)
    for profile in profiles:
        # Preserve explicit permanent observations omitted by precise classification.
        # Existing choices and chronology remain authoritative, including exclusions.
        covered = {key for field in ('global_traits', 'temporary_states', 'unresolved_conflicts')
                   for row in profile[field] for key in row['evidence_ids']}
        covered.update(row[key] for row in profile['chronological_changes']
                       for key in ('before_id', 'after_id'))
        covered.update(ref['evidence_id'] for row in profile['trait_relations'] for ref in row['evidence'])
        permanent = [row for row in profile['attributes']['stable_physical']
                     if row['evidence_id'] not in covered and row['classification'] == 'explicit'
                     and row['persistence'] == 'persistent'
                     and row['attribute'] in ('age', 'physical_appearance', 'distinguishing_feature')]
        conflicts = {rows[0]['trait'] for rows in profile['linked_trait_conflicts']}
        evolving = {row['trait'] for row in profile['chronological_changes']}
        profile['ignored_traits'] = sorted(set(profile['ignored_traits']) | conflicts)
        profile['blocked_fields'] = sorted(set(profile['blocked_fields']) |
                                          {selection.field_for(t) for t in conflicts | evolving})
        profile['effective_traits'] = [r for r in profile['effective_traits']
            if r['trait'] not in conflicts | evolving | {'clothing', 'accessory', 'injury', 'context'}]
        profile['effective_traits'].extend({'trait': row['attribute'], 'value': row['value'],
                                            'evidence_ids': [row['evidence_id']]} for row in permanent)
        if permanent:
            # No precise field was assigned: prevent invented defaults from
            # contradicting any of these source-supported physical observations.
            profile['blocked_fields'] = sorted(set(profile['blocked_fields']) | set(PHYSICAL))
    return profiles


def select_profiles(data, paths):
    source_key = character_evidence.source_key
    keys = {source_key(path) for path in paths}
    chapters = data['source_document']['chapters']
    for key in keys:
        if sum(source_key(c['source_file']) == key for c in chapters) != 1:
            raise ValueError(f'Selected source does not identify exactly one evidence chapter: {key}. Regenerate character evidence.')
    if not keys:
        raise ValueError('Select at least one chapter.')
    return [p for p in prepare_profiles(data) if any(
        source_key(r['observation']['provenance']['file']) in keys for r in p['supporting_evidence'])]


def source_facts(profile):
    return {'stable_visual_description': '; '.join(dict.fromkeys(r['value'] for r in profile['effective_traits'])),
            'distinguishing_features': []}


def generate(data, paths, output, previous=None):
    validate_profiles(data)
    profiles = select_profiles(data, paths)
    if previous is not None:
        util.require_schema(previous, visual_designs.DESIGN_SCHEMA_VERSION)
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
            'exclude_image': old.get('exclude_image', False),
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
    result = {'format_version': VERSION, 'profiles': profiles, 'visual_designs': designs}
    saved = output / 'character_visual_profiles.json'
    util.save_json(saved, result)
    ambiguous = [p['local_id'] for p in profiles if p['identity_status'] == 'ambiguous']
    report = {'status': 'incomplete' if ambiguous else 'complete', 'ambiguous_profile_ids': ambiguous}
    without_traits = [e['character_id'] for e in entities
                      if not e['source_facts']['stable_visual_description'] and not e['added_details']]
    if without_traits:
        report['profiles_without_visual_traits'] = without_traits
        report['reason'] = 'No stable physical traits or edited visual details were available for these profiles.'
    if not entities:
        report['empty_reason'] = 'The selected chapters contain no usable character profiles.'
    util.save_json(output / 'character_visual_profiles.diagnostics.json', report)
    return result, str(saved)
