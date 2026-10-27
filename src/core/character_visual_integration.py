"""Attach chapter enrichment to registry identities without altering asset order."""
from copy import deepcopy
import json
import warnings

from . import character_chapter_enrichment as enrichment
from .editable_schemas import SCHEMA_DIR, DESIGN_SCHEMA, validate_document


def validate_input(data, chapters):
    if data is None:
        return
    if data.get('format_version') != 3:
        raise ValueError('Use current Character Visual Profile Generator output.')
    local = data.get('chapter_enrichment')
    contract = json.loads((SCHEMA_DIR / 'character_chapter_enrichment.schema.json').read_text(encoding='utf-8'))
    validate_document(local, contract, 'character_chapter_enrichment.json')
    validate_document(data['visual_designs'], DESIGN_SCHEMA, 'visual_designs.json')
    matches = [c for c in chapters if enrichment.source_key(c.get('source_file') or c.get('source', {}).get('file', ''))
               == enrichment.source_key(local['source_file'])]
    if len(matches) != 1 or enrichment.timeline_digest(matches[0]) != local['chapter_digest']:
        raise ValueError('Character enrichment belongs to a different source or cinematic sequence revision.')
    ids = {p['local_id'] for p in data['profiles']}
    sequences = {s['sequence']: s for s in matches[0]['sequences']}
    if [s['sequence'] for s in local['sequences']] != list(sequences):
        raise ValueError('Enrichment must cover every cinematic sequence once, in order.')
    for seq in local['sequences']:
        source = sequences[seq['sequence']]['source']
        for row in seq['characters']:
            if row['presence'] != 'uncertain' and row['local_id'] not in ids:
                raise ValueError('Enrichment references an unknown character profile.')
        for row in seq['characters'] + seq['states'] + seq['unresolved']:
            if row['source_quote'] not in source:
                raise ValueError('Enrichment quote does not match its cinematic sequence.')
        for row in seq['states']:
            evidence_ids = [r['evidence_id'] for r in row['evidence']]
            if (row['local_id'] not in ids or len(evidence_ids) != len(set(evidence_ids))
                    or set(row['evidence_ids']) != set(evidence_ids)):
                raise ValueError('Local state identity and evidence references must resolve uniquely.')


def apply(data, chapters, registry):
    if data is None:
        return {}
    validate_input(data, chapters)
    local = data['chapter_enrichment']
    chapter = next(c for c in chapters if enrichment.source_key(c.get('source_file') or c.get('source', {}).get('file', ''))
                   == enrichment.source_key(local['source_file']))
    cid = chapter['chapter_id']
    designs = {d['global_id']: d for d in data['visual_designs']['entities']}
    def names(item):
        return {n.casefold().strip() for n in [item['canonical_name'], *item.get('aliases', [])] if n.strip()}
    used, mapped = set(), {}
    for entity in registry:
        if entity['entity_type'] != 'character' or cid not in entity.get('chapters_seen', []):
            continue
        candidates = [p for p in data['profiles'] if names(p) & names(entity)]
        if len(candidates) != 1 or candidates[0]['local_id'] in used:
            warnings.warn(f"Character enrichment: unresolved registry mapping for {entity['canonical_name']}; no arbitrary merge.",
                          UserWarning, stacklevel=2)
            continue
        profile = candidates[0]
        lid = profile['local_id']
        if profile['identity_status'] == 'ambiguous' and len(profile['linked_local_ids']) == 1:
            warnings.warn(f"Character enrichment: ambiguous identity {lid} requires review.", UserWarning, stacklevel=2)
            continue
        used.add(lid)
        mapped[lid] = entity['global_id']
        entity['character_id'] = lid
        entity['stable_visual_description'] = designs[lid]['source_facts']['stable_visual_description']
        entity['distinguishing_features'] = designs[lid]['source_facts']['distinguishing_features']
        entity['ignored_character_traits'] = deepcopy(profile['ignored_traits'])
        entity['character_enrichment'] = {cid: {'global_traits': deepcopy(profile['effective_traits']),
            'ignored_traits': deepcopy(profile['ignored_traits']), 'sequences': [
                {'sequence': s['sequence'], 'presence': [r for r in s['characters'] if r['local_id'] == lid],
                 'states': [r for r in s['states'] if r['local_id'] == lid]} for s in local['sequences']]}}
    return {cid: {**deepcopy(local), 'registry_ids': mapped,
                  'ignored_traits': {mapped[p['local_id']]: p['ignored_traits'] for p in data['profiles']
                                     if p['local_id'] in mapped}}}


def sequence_context(refs, chapter_id, sequence):
    local = refs.get('character_enrichment', {}).get(chapter_id)
    if not local:
        return None
    rows = next((s for s in local['sequences'] if s['sequence'] == sequence), None)
    if rows is not None:
        rows = {**deepcopy(rows), 'characters': [r for r in rows['characters'] if r['local_id'] in local['registry_ids']],
                'states': [r for r in rows['states'] if r['local_id'] in local['registry_ids']]}
    return {'instruction': 'Use global source facts and approved defaults for identity only. '
            'Apply these states only within this sequence and at their attested narrative time. '
            'States override defaults. Offscreen and mentioned people must not become visible. '
            'Never invent ignored traits or propagate temporary states across sequences.',
            'registry_ids': local['registry_ids'], 'ignored_traits': local['ignored_traits'], 'sequence': rows}
