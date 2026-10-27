"""Project editable identities and precise choices without changing their sources."""
from copy import deepcopy

from .character_profile_document import validate_document


def field_for(trait):
    if trait.startswith('eye_'):
        return 'eyes'
    if trait.startswith('hair_') or trait in ('facial_hair', 'body_hair'):
        return 'hair'
    if trait.startswith('skin_'):
        return 'skin'
    return {'age': 'age_appearance', 'clothing': 'default_outfit',
            'accessory': 'accessories'}.get(trait, 'build')


def project(data):
    validate_document(data)
    principals = {p['local_id']: deepcopy(p) for p in data['profiles'] if p['link_to_local_id'] is None}
    for p in principals.values():
        p['linked_local_ids'] = [p['local_id']]
    for alias in data['profiles']:
        if alias['link_to_local_id'] is None:
            continue
        target = principals[alias['link_to_local_id']]
        target['linked_local_ids'].append(alias['local_id'])
        target['aliases'] = list(dict.fromkeys([*target['aliases'], alias['canonical_name'], *alias['aliases']]))
        for key in ('supporting_evidence', 'global_traits', 'temporary_states', 'unresolved_conflicts',
                    'chronological_changes', 'trait_relations', 'identity_support'):
            target[key].extend(deepcopy(alias[key]))
        for key in target['attributes']:
            target['attributes'][key].extend(deepcopy(alias['attributes'][key]))
    for p in principals.values():
        records = {r['evidence_id']: r['observation'] for r in p['supporting_evidence']}
        ignored, selected = [], []
        conflict_traits = {c['trait'] for c in p['unresolved_conflicts']}
        for c in p['unresolved_conflicts']:
            choice = next((x for x in c['choices'] if x['selected']), None)
            if choice is None:
                ignored.append(c['trait'])
            else:
                selected.append({'trait': c['trait'], 'value': choice['value'],
                                 'evidence_ids': choice['evidence_ids'], 'scope': c['scope']})
        p['ignored_traits'] = list(dict.fromkeys(ignored))
        p['selected_traits'] = [r for r in selected if r['trait'] not in ignored]
        evolved = {c[k] for c in p['chronological_changes'] for k in ('before_id', 'after_id')}
        p['effective_traits'] = [deepcopy(r) for r in p['global_traits']
                                 if r['trait'] not in conflict_traits and not set(r['evidence_ids']) & evolved]
        # A selected temporal value remains local. It never becomes a retroactive trait.
        for r in p['selected_traits']:
            if not set(r['evidence_ids']) & evolved and all(
                    records[k]['classification'] == 'explicit' and records[k]['persistence'] == 'persistent'
                    and records[k]['attribute'] in ('age', 'physical_appearance', 'distinguishing_feature')
                    for k in r['evidence_ids']):
                p['effective_traits'].append(deepcopy(r))
        # Newly linked profiles can carry incompatible decisions that were never compared.
        # Keep them out of reusable canon until the bounded comparison has resolved them.
        p['linked_trait_conflicts'] = []
        if len(p['linked_local_ids']) > 1:
            by_trait = {}
            for row in p['effective_traits']:
                by_trait.setdefault(row['trait'], []).append(row)
            p['linked_trait_conflicts'] = [rows for rows in by_trait.values()
                                           if len({r['value'].casefold() for r in rows}) > 1]
        p['blocked_fields'] = sorted({field_for(t) for t in ignored})
    return list(principals.values())
