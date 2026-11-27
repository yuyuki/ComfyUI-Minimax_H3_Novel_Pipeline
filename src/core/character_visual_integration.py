"""Join validated physical profiles to consolidated character identities."""
from collections import defaultdict
from copy import deepcopy
import warnings
from . import character_evidence, util, visual_designs

from . import character_visual_profiles as visual
from .character_profile_document import validate_document as validate_profiles
from .character_visual_selection import field_for
from .editable_schemas import DESIGN_SCHEMA, validate_document
from .lmstudio_pipeline import comfy_interrupt_check


def validate_input(data):
    if data is None:
        return
    if not isinstance(data, dict) or data.get('format_version') != visual.VERSION:
        raise ValueError('Unsupported character visual profiles format. Regenerate with Character Visual Profile Generator.')
    if not isinstance(data.get('visual_designs'), dict):
        raise ValueError('Character visual profiles require a visual_designs object.')
    character_evidence.reject_embedded_source(data)
    util.require_schema(data['visual_designs'], visual_designs.DESIGN_SCHEMA_VERSION)
    validate_document(data['visual_designs'], DESIGN_SCHEMA, 'visual_designs.json')
    validate_profiles(data['visual_designs'].get('character_profiles'))


def _names(item):
    return {name.strip().casefold() for name in
            [item['canonical_name'], *item.get('aliases', [])] if name.strip()}


def apply(designs, chapters, registry, diagnostics):
    if not designs or 'character_profiles' not in designs:
        return {}
    util.require_schema(designs, visual_designs.DESIGN_SCHEMA_VERSION)
    validate_document(designs, DESIGN_SCHEMA, 'visual_designs.json')
    document = designs['character_profiles']
    sources = document['source_document']
    character_evidence.validate_document(sources)
    sources = {character_evidence.source_key(c['source_file']): c for c in sources['chapters']}
    texts = {key: character_evidence.read_verified_source(chapter) for key, chapter in sources.items()}
    validate_profiles(document, texts=texts)
    for chapter in chapters:
        comfy_interrupt_check()
        key = character_evidence.source_key(chapter.get('source_file') or chapter.get('source', {}).get('file', ''))
        if key not in texts:
            raise ValueError(f'Cinematic chapter has no matching character source: {key}. Regenerate character evidence for this chapter.')
        text = texts[key]
        cursor = 0
        for sequence in chapter['sequences']:
            excerpt = sequence['source']
            start = text.find(excerpt, cursor)
            if not excerpt.strip() or start < 0 or (text[cursor:start].strip() and text.find(excerpt, start + 1) >= 0):
                raise ValueError(f"Cinematic source does not match a unique ordered passage: sequence {sequence['sequence']}.")
            cursor = start + len(excerpt)

    profiles = visual.prepare_profiles(document)
    items = {d['character_id']: d for d in designs['entities'] if d['entity_type'] == 'character' and 'character_id' in d}
    if len(items) != sum('character_id' in d for d in designs['entities'] if d['entity_type'] == 'character'):
        raise ValueError('Duplicate character identity in visual_designs.json.')
    by_id = {p['local_id']: p for p in profiles}
    if not set(items) <= by_id.keys():
        raise ValueError('Unknown character identity in visual_designs.json.')
    ids_by_name = defaultdict(set)
    for lid, profile in by_id.items():
        for name in _names(profile):
            ids_by_name[name].add(lid)
    matches_by_entity = {}
    entities_by_profile = defaultdict(list)
    for entity in registry:
        if entity['entity_type'] != 'character':
            continue
        gid = entity['global_id']
        matches = set().union(*(ids_by_name.get(name, ()) for name in _names(entity)))
        matches_by_entity[gid] = matches
        for lid in matches:
            entities_by_profile[lid].append(gid)

    imported = {}
    for entity in registry:
        comfy_interrupt_check()
        gid = entity['global_id']
        matches = matches_by_entity.get(gid)
        if not matches:
            continue
        lid = min(matches)
        profile = by_id[lid]
        item = items.get(lid, {'character_id': lid, 'identity_status': profile['identity_status'], 'added_details': {}})
        if (len(matches) != 1 or len(entities_by_profile[lid]) != 1
                or 'ambiguous' in (profile['identity_status'], item['identity_status'])):
            message = 'Ambiguous character registry mapping; no facts imported.'
            warnings.warn(message, UserWarning, stacklevel=2)
            diagnostics.append({'stage': 'character_registry_identity', 'status': 'incomplete',
                                'chapter_ids': entity.get('chapters_seen', []), 'entity_id': gid,
                                'profile_ids': sorted(matches), 'message': message})
            continue
        facts = visual.source_facts(profile)
        entity.update(character_id=lid, **deepcopy(facts), ignored_character_traits=deepcopy(profile['ignored_traits']))
        blocked = set(profile['blocked_fields']) | {field_for(r['trait']) for r in profile['effective_traits']}
        approved = {k: v for k, v in item['added_details'].items()
                    if k in visual_designs.DESIGN_TRAITS['character'] and k not in blocked}
        imported[gid] = {**deepcopy(item), 'source_facts': facts, 'added_details': approved,
                         'blocked_design_fields': sorted(blocked)}
    return imported
