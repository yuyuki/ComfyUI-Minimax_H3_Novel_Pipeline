"""Second-pass identity repair of generated designs, preserving editable facts."""
from copy import deepcopy
import re

from . import character_profile_batches as batches, character_profiles as source, util
from .character_profile_document import validate_document


def candidates(designs):
    """Nominate overlapping names, never use them as proof of identity."""
    document = designs['character_profiles']
    selected = {e['character_id'] for e in designs['entities'] if 'character_id' in e}
    # Explicit operator links are authoritative and are not reconsidered here.
    linked = {p['link_to_local_id'] for p in document['profiles'] if p['link_to_local_id']}
    profiles = [p for p in document['profiles'] if p['link_to_local_id'] is None and p['local_id'] not in linked]
    names = {p['local_id']: {source._normalized(n).casefold() for n in
             [p['canonical_name'], *p['aliases']] if n.strip()} for p in profiles}
    edges = []
    for i, left in enumerate(profiles):
        for right in profiles[i + 1:]:
            if any(re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', b)
                   or re.search(r'(?<!\w)' + re.escape(b) + r'(?!\w)', a)
                   for a in names[left['local_id']] for b in names[right['local_id']]):
                edges.append({left['local_id'], right['local_id']})
    active = set(selected)
    while True:
        expanded = active | set().union(*(edge for edge in edges if edge & active))
        if expanded == active:
            break
        active = expanded
    nominated = set().union(*(edge for edge in edges if edge <= active))
    return [p for p in profiles if p['local_id'] in nominated]


def repair(designs, profiles, output, client, model, args):
    """Return a validated copy and defaults for deterministic reprojection."""
    corrected = deepcopy(designs)
    document = corrected['character_profiles']
    records = source.prepare_evidence(document['source_document'])
    groups, usable, owners = [], {}, {}
    report = {'links': [], 'unresolved': []}
    for profile in profiles:
        ids = [r['evidence_id'] for r in profile['supporting_evidence']]
        group = {k: deepcopy(profile[k]) for k in ('canonical_name', 'identity_status')}
        group.update(evidence=[{'evidence_id': k} for k in ids],
                     identity_support=[{'evidence_id': k} for k in profile['identity_support']])
        subset = {k: records[k] for k in ids}
        try:
            source.build_profiles({'profiles': [group]}, subset)
        except ValueError as exc:
            # Do not reinterpret an operator's renamed/reviewed identity.
            report['unresolved'].append({'local_ids': [profile['local_id']], 'reason': str(exc)})
            continue
        groups.append(group)
        usable.update(subset)
        owners.update(dict.fromkeys(ids, profile['local_id']))
    requests = batches.Requests(output, client, model, args)
    try:
        identities = batches.reconcile_identities(groups, usable, requests) if len(groups) > 1 else []
    finally:
        util.save_json(output / 'character_visual_identity_requests.json', requests.stats)
    original = {p['local_id']: p for p in document['profiles']}
    entities = {e['character_id']: e for e in corrected['entities'] if 'character_id' in e}
    for identity in identities:
        members = {owners[r['evidence_id']] for r in identity['supporting_evidence']}
        if len(members) < 2:
            continue
        defaults = {}
        conflict = False
        for lid in sorted(members):
            for field, value in entities.get(lid, {}).get('added_details', {}).items():
                if field in defaults and defaults[field] != value:
                    conflict = True
                defaults[field] = value
        if conflict:
            report['unresolved'].append({'local_ids': sorted(members),
                                        'reason': 'Conflicting edited physical defaults; identities left unchanged.'})
            continue
        # Keep an already exported identity stable for the selected chapters.
        principal = min(members, key=lambda lid: (lid not in entities,
                        original[lid]['canonical_name'] != identity['canonical_name'], lid))
        original[principal]['identity_status'] = 'confirmed'
        for lid in sorted(members - {principal}):
            original[lid]['link_to_local_id'] = principal
            report['links'].append({'local_id': lid, 'link_to_local_id': principal,
                                    'evidence_ids': identity['identity_support']})
        if principal in entities:
            entities[principal]['added_details'] = defaults
    linked_ids = {link['local_id'] for link in report['links']}
    for profile in profiles:
        if profile['local_id'] not in linked_ids and not any(
                link['link_to_local_id'] == profile['local_id'] for link in report['links']):
            report['unresolved'].append({'local_ids': [profile['local_id']],
                                        'reason': 'No new unique source-supported identity link.'})
    validate_document(document)
    util.save_json(output / 'character_visual_identity_review.json', report)
    return corrected
