#!/usr/bin/env python3
"""Consolidate chapter lore JSONs into a scalable book/series knowledge graph."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import novel_pipeline_common as npc

SCRIPT_VERSION='1.0.0'
INPUT_SCHEMA='novel-lore.chapter.v1'
OUTPUT_SCHEMA='novel-lore.graph.v1'
TYPE_PREFIX={'character':'CHAR','location':'LOC','object':'OBJ','artifact':'ART','organization':'ORG','people':'PEO','species':'SPEC','language':'LANG','concept':'CON','custom':'CUS','other':'ENT'}

RECON_ITEM={
 'local_id':{'type':'string'},'match_global_id':{'type':'string'},'canonical_name':{'type':'string'},
 'aliases':{'type':'array','items':{'type':'string'}},'summary':{'type':'string'},
 'reason':{'type':'string'},'confidence':{'type':'number','minimum':0,'maximum':1}
}
RECON_SCHEMA={'name':'lore_entity_reconciliation_v1','strict':True,'schema':{'type':'object','properties':{'resolutions':{'type':'array','items':{'type':'object','properties':RECON_ITEM,'required':list(RECON_ITEM),'additionalProperties':False}}},'required':['resolutions'],'additionalProperties':False}}
AUDIT_ITEM={'keep_global_id':{'type':'string'},'merge_global_ids':{'type':'array','items':{'type':'string'}},'canonical_name':{'type':'string'},'aliases':{'type':'array','items':{'type':'string'}},'summary':{'type':'string'},'reason':{'type':'string'}}
AUDIT_SCHEMA={'name':'lore_cluster_audit_v1','strict':True,'schema':{'type':'object','properties':{'merge_groups':{'type':'array','items':{'type':'object','properties':AUDIT_ITEM,'required':list(AUDIT_ITEM),'additionalProperties':False}}},'required':['merge_groups'],'additionalProperties':False}}

RECON_SYSTEM=r'''
Reconcile chapter-local lore entities against an existing global registry.
For every incoming local_id, choose exactly one supplied candidate global_id or NEW.
Merge only if they clearly denote the exact same fictional/world entity. Names and
aliases are strong evidence, but titles, relationships, role, species/type and summary
must remain compatible. Never merge merely similar people/places/things. When unsure,
choose NEW. Canonicalization may improve names/aliases/summary but must not invent lore.
'''.strip()
AUDIT_SYSTEM=r'''
Audit one SMALL candidate cluster for accidental duplicate global lore entities.
Return merge groups only for exact same entities. Never merge merely similar names or
roles and never merge across types. Prefer the earliest/most established global ID as
keep_global_id when otherwise equivalent. Preserve aliases and source-grounded summary.
'''.strip()


def similarity(a:dict[str,Any],b:dict[str,Any])->float:
    an=[npc.norm_name(a.get('canonical_name','')),*[npc.norm_name(x) for x in a.get('aliases',[])]]
    bn=[npc.norm_name(b.get('canonical_name','')),*[npc.norm_name(x) for x in b.get('aliases',[])]]
    best=0.0
    for x in filter(None,an):
        for y in filter(None,bn):
            if x==y:return 1.0
            ratio=difflib.SequenceMatcher(None,x,y).ratio(); tx=set(x.split()); ty=set(y.split()); overlap=len(tx&ty)/max(1,len(tx|ty))
            best=max(best,ratio,overlap)
    return best


def compact_entity(e:dict[str,Any])->dict[str,Any]:
    return {k:e.get(k) for k in ('global_id','entity_type','canonical_name','aliases','summary','stable_traits','importance','visual_global_id')}


def load_chapters(paths:list[Path])->list[dict[str,Any]]:
    out=[]
    for p in paths:
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('schema_version')!=INPUT_SCHEMA: raise ValueError(f'{p}: expected {INPUT_SCHEMA}, got {d.get("schema_version")!r}')
        d['_path']=str(p.resolve()); out.append(d)
    return out



def seed_visual_registry(path:Path|None)->list[dict[str,Any]]:
    """Seed character/location/object identities from Step 2V so lore IDs align with H3 visual IDs."""
    if path is None:return []
    data=json.loads(path.read_text(encoding='utf-8')); out=[]
    for v in data.get('entities',[]):
        if v.get('entity_type') not in {'character','location','object'}:continue
        out.append({'global_id':v['global_id'],'visual_global_id':v['global_id'],'entity_type':v['entity_type'],
                    'canonical_name':v.get('canonical_name',v['global_id']),'aliases':list(v.get('aliases',[])),
                    'summary':v.get('stable_visual_description',''),'stable_traits':list(v.get('distinguishing_features',[])),
                    'importance':v.get('importance','minor'),'chapters_seen':list(v.get('chapters_seen',[])),
                    'source_entities':[],'evidence':[]})
    return out


def build_chapter_entity_map(registry:list[dict[str,Any]])->dict[str,dict[str,str]]:
    out={}
    for e in registry:
        for src in e.get('source_entities',[]):
            cid,lid=src.get('chapter_id',''),src.get('local_id','')
            if cid and lid:out.setdefault(cid,{})[lid]=e['global_id']
    return out


def build_indexes(graph:dict[str,Any],chapter_order:list[str])->dict[str,Any]:
    entity_fact={e['global_id']:[] for e in graph['entities']}; entity_rel={e['global_id']:[] for e in graph['entities']}; entity_event={e['global_id']:[] for e in graph['entities']}
    chapter_fact={c:[] for c in chapter_order};chapter_rel={c:[] for c in chapter_order};chapter_event={c:[] for c in chapter_order};chapter_term={c:[] for c in chapter_order}
    for f in graph['facts']:
        for c in f.get('chapters',[]):chapter_fact.setdefault(c,[]).append(f['fact_id'])
        for gid in {f.get('subject_global_id',''),f.get('object_global_id','')}:
            if gid:entity_fact.setdefault(gid,[]).append(f['fact_id'])
    for r in graph['relationships']:
        for c in r.get('chapters',[]):chapter_rel.setdefault(c,[]).append(r['relationship_id'])
        for gid in {r.get('subject_global_id',''),r.get('object_global_id','')}:
            if gid:entity_rel.setdefault(gid,[]).append(r['relationship_id'])
    for e in graph['events']:
        cid=e.get('chapter_id','');chapter_event.setdefault(cid,[]).append(e['event_id'])
        gids={e.get('location_global_id','')}|{x.get('global_id','') for x in e.get('participants',[])}
        for gid in gids:
            if gid:entity_event.setdefault(gid,[]).append(e['event_id'])
    for t in graph['terminology']:
        for c in t.get('chapters',[]):chapter_term.setdefault(c,[]).append(t['term_id'])
    return {'entity_fact_ids':entity_fact,'entity_relationship_ids':entity_rel,'entity_event_ids':entity_event,
            'chapter_fact_ids':chapter_fact,'chapter_relationship_ids':chapter_rel,'chapter_event_ids':chapter_event,'chapter_term_ids':chapter_term}


def potential_conflicts(facts:list[dict[str,Any]])->list[dict[str,Any]]:
    buckets={}
    for f in facts:
        subj=f.get('subject_global_id') or npc.norm_name(f.get('subject_name',''))
        key=(subj,npc.norm_name(f.get('predicate','')),npc.norm_name(f.get('temporal_scope','')))
        obj=f.get('object_global_id') or npc.norm_name(f.get('object_name') or f.get('object_text',''))
        if subj and key[1] and obj:buckets.setdefault(key,{}).setdefault(obj,[]).append(f.get('fact_id',''))
    return [{'subject':k[0],'predicate':k[1],'temporal_scope':k[2],'alternatives':[{'object':o,'fact_ids':ids} for o,ids in vals.items()]}
            for k,vals in buckets.items() if len(vals)>1]

def next_gid(registry:list[dict[str,Any]],typ:str)->str:
    pref=TYPE_PREFIX.get(typ,'ENT'); nums=[]
    for e in registry:
        m=re.fullmatch(rf'{re.escape(pref)}_(\d+)',e['global_id'])
        if m: nums.append(int(m.group(1)))
    return f'{pref}_{(max(nums)+1 if nums else 1):04d}'


def candidates_for(item:dict[str,Any],registry:list[dict[str,Any]],top_k:int,include_all_below:int)->list[dict[str,Any]]:
    pool=[e for e in registry if e['entity_type']==item['entity_type']]
    if len(pool)<=include_all_below:return [compact_entity(e) for e in pool]
    scored=sorted(((similarity(item,e),e) for e in pool),key=lambda x:x[0],reverse=True)
    sel=[e for s,e in scored[:top_k] if s>=.14] or [e for _,e in scored[:min(3,len(scored))]]
    return [compact_entity(e) for e in sel]


def reconcile_chapter(client,model,chapter:dict[str,Any],registry:list[dict[str,Any]],args)->dict[str,str]:
    incoming=chapter.get('entities',[])
    cand={e['local_id']:candidates_for(e,registry,args.candidate_count,args.include_all_below) for e in incoming}
    result=npc.chat_json(client,model,RECON_SYSTEM,
        f"Chapter: {chapter['chapter_id']}\n\nINCOMING ENTITIES:\n{json.dumps(incoming,ensure_ascii=False,indent=2)}\n\nCANDIDATES BY LOCAL ID:\n{json.dumps(cand,ensure_ascii=False,indent=2)}",
        RECON_SCHEMA,args.temperature,args.max_tokens)
    resolutions={r['local_id']:r for r in result.get('resolutions',[])}; mapping={}; by_id={e['global_id']:e for e in registry}
    for item in incoming:
        r=resolutions.get(item['local_id'],{}); gid=r.get('match_global_id','NEW')
        allowed={c['global_id'] for c in cand.get(item['local_id'],[])}
        if gid!='NEW' and (gid not in by_id or gid not in allowed): gid='NEW'
        if gid=='NEW':
            gid=next_gid(registry,item.get('entity_type','other'))
            e={'global_id':gid,'visual_global_id':'','entity_type':item.get('entity_type','other'),'canonical_name':(r.get('canonical_name') or item.get('canonical_name','')).strip(),
               'aliases':npc.dedupe(item.get('aliases',[])+r.get('aliases',[]),60),'summary':(r.get('summary') or item.get('summary','')).strip(),
               'stable_traits':npc.dedupe(item.get('stable_traits',[]),40),'importance':item.get('importance','minor'),'chapters_seen':[chapter['chapter_id']],
               'source_entities':[{'chapter_id':chapter['chapter_id'],'local_id':item['local_id']}], 'evidence':npc.dedupe(item.get('evidence',[]),20)}
            registry.append(e); by_id[gid]=e
        else:
            e=by_id[gid]; e['canonical_name']=(r.get('canonical_name') or e['canonical_name']).strip(); e['aliases']=npc.dedupe(e.get('aliases',[])+item.get('aliases',[])+r.get('aliases',[]),60)
            if r.get('summary'): e['summary']=r['summary'].strip()
            elif item.get('summary') and len(item['summary'])>len(e.get('summary','')): e['summary']=item['summary'].strip()
            e['stable_traits']=npc.dedupe(e.get('stable_traits',[])+item.get('stable_traits',[]),40); e['evidence']=npc.dedupe(e.get('evidence',[])+item.get('evidence',[]),20)
            if chapter['chapter_id'] not in e['chapters_seen']: e['chapters_seen'].append(chapter['chapter_id'])
            src={'chapter_id':chapter['chapter_id'],'local_id':item['local_id']}
            if src not in e['source_entities']: e['source_entities'].append(src)
        mapping[item['local_id']]=gid
    return mapping


def chapter_name_map(chapter:dict[str,Any],local_to_gid:dict[str,str])->dict[str,str]:
    out={}
    for e in chapter.get('entities',[]):
        gid=local_to_gid.get(e['local_id'])
        if not gid: continue
        for n in [e.get('canonical_name',''),*e.get('aliases',[])]:
            k=npc.norm_name(n)
            if k: out[k]=gid
    return out


def resolve_name(name:str,cmap:dict[str,str],registry:list[dict[str,Any]])->str|None:
    k=npc.norm_name(name)
    if not k:return None
    if k in cmap:return cmap[k]
    hits=[]
    for e in registry:
        names=[e.get('canonical_name',''),*e.get('aliases',[])]
        if any(npc.norm_name(n)==k for n in names): hits.append(e['global_id'])
    return hits[0] if len(set(hits))==1 else None


def add_chapter_graph(chapter:dict[str,Any],mapping:dict[str,str],registry:list[dict[str,Any]],graph:dict[str,Any])->None:
    cmap=chapter_name_map(chapter,mapping); cid=chapter['chapter_id']
    graph['chapter_summaries'][cid]=chapter.get('chapter_summary','')
    def evidence(x): return [{'chapter_id':cid,'anchor':a} for a in npc.dedupe(x,4)]

    for f in chapter.get('facts',[]):
        sg=resolve_name(f.get('subject_name',''),cmap,registry); og=resolve_name(f.get('object_name',''),cmap,registry)
        rec={'fact_id':'','subject_global_id':sg or '','subject_name':f.get('subject_name',''),'predicate':f.get('predicate',''),
             'object_global_id':og or '','object_name':f.get('object_name',''),'object_text':f.get('object_text',''),'qualifiers':npc.dedupe(f.get('qualifiers',[]),12),
             'temporal_scope':f.get('temporal_scope',''),'confidence':float(f.get('confidence',.8)),'chapters':[cid],'evidence':evidence(f.get('evidence',[]))}
        key=(sg or npc.norm_name(rec['subject_name']),npc.norm_name(rec['predicate']),og or npc.norm_name(rec['object_name'] or rec['object_text']),npc.norm_name(rec['temporal_scope']))
        old=graph['_fact_index'].get(key)
        if old:
            old['qualifiers']=npc.dedupe(old['qualifiers']+rec['qualifiers'],16); old['evidence']=(old['evidence']+rec['evidence'])[:20]; old['confidence']=max(old['confidence'],rec['confidence'])
            if cid not in old['chapters']: old['chapters'].append(cid)
        else:
            rec['fact_id']=f"FACT_{len(graph['facts'])+1:06d}"; graph['facts'].append(rec); graph['_fact_index'][key]=rec
        if not sg or (rec['object_name'] and not og): graph['unresolved_mentions'].append({'chapter_id':cid,'kind':'fact','subject_name':rec['subject_name'] if not sg else '','object_name':rec['object_name'] if rec['object_name'] and not og else ''})

    for r in chapter.get('relationships',[]):
        sg=resolve_name(r.get('subject_name',''),cmap,registry); og=resolve_name(r.get('object_name',''),cmap,registry)
        rec={'relationship_id':'','subject_global_id':sg or '','subject_name':r.get('subject_name',''),'relation':r.get('relation',''),'object_global_id':og or '',
             'object_name':r.get('object_name',''),'state':r.get('state',''),'confidence':float(r.get('confidence',.8)),'chapters':[cid],'evidence':evidence(r.get('evidence',[]))}
        key=(sg or npc.norm_name(rec['subject_name']),npc.norm_name(rec['relation']),og or npc.norm_name(rec['object_name']),npc.norm_name(rec['state']))
        old=graph['_rel_index'].get(key)
        if old:
            old['evidence']=(old['evidence']+rec['evidence'])[:20]; old['confidence']=max(old['confidence'],rec['confidence'])
            if cid not in old['chapters']: old['chapters'].append(cid)
        else:
            rec['relationship_id']=f"REL_{len(graph['relationships'])+1:06d}"; graph['relationships'].append(rec); graph['_rel_index'][key]=rec
        if not sg or not og: graph['unresolved_mentions'].append({'chapter_id':cid,'kind':'relationship','subject_name':rec['subject_name'] if not sg else '','object_name':rec['object_name'] if not og else ''})

    for ev in chapter.get('events',[]):
        participants=[]
        for n in ev.get('participants',[]): participants.append({'name':n,'global_id':resolve_name(n,cmap,registry) or ''})
        loc=resolve_name(ev.get('location_name',''),cmap,registry)
        graph['events'].append({'event_id':f"EVT_{len(graph['events'])+1:06d}",'chapter_id':cid,'title':ev.get('title',''),'summary':ev.get('summary',''),
            'participants':participants,'location_name':ev.get('location_name',''),'location_global_id':loc or '','sequence_hint':ev.get('sequence_hint',''),
            'causes':npc.dedupe(ev.get('causes',[]),8),'consequences':npc.dedupe(ev.get('consequences',[]),8),'evidence':evidence(ev.get('evidence',[]))})

    for t in chapter.get('terminology',[]):
        k=npc.norm_name(t.get('term',''))
        if not k: continue
        old=graph['_term_index'].get(k)
        rec={'term_id':'','term':t.get('term',''),'definition':t.get('definition',''),'category':t.get('category',''),'aliases':npc.dedupe(t.get('aliases',[]),12),'chapters':[cid],'evidence':evidence(t.get('evidence',[]))}
        if old:
            if len(rec['definition'])>len(old['definition']): old['definition']=rec['definition']
            old['aliases']=npc.dedupe(old['aliases']+rec['aliases'],20); old['evidence']=(old['evidence']+rec['evidence'])[:20]
            if cid not in old['chapters']: old['chapters'].append(cid)
        else:
            rec['term_id']=f"TERM_{len(graph['terminology'])+1:05d}"; graph['terminology'].append(rec); graph['_term_index'][k]=rec


def cluster_entities(registry:list[dict[str,Any]],threshold:float,max_size:int)->list[list[str]]:
    by_type={}
    for e in registry: by_type.setdefault(e['entity_type'],[]).append(e)
    clusters=[]
    for typ,items in by_type.items():
        # Blocking: compare only entities sharing a normalized token/prefix, which keeps this scalable.
        buckets={}
        for i,e in enumerate(items):
            names=[npc.norm_name(e.get('canonical_name','')),*[npc.norm_name(x) for x in e.get('aliases',[])]]
            keys=set()
            for n in filter(None,names):
                toks=n.split(); keys.update('t:'+t for t in toks if len(t)>=3); keys.add('p:'+n[:3])
            for k in keys: buckets.setdefault(k,set()).add(i)
        parent=list(range(len(items)))
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra
        seen=set()
        for ids in buckets.values():
            ids=list(ids)
            for ai in range(len(ids)):
                for bi in range(ai+1,len(ids)):
                    a,b=ids[ai],ids[bi]; pair=(min(a,b),max(a,b))
                    if pair in seen: continue
                    seen.add(pair)
                    if similarity(items[a],items[b])>=threshold: union(a,b)
        comps={}
        for i in range(len(items)): comps.setdefault(find(i),[]).append(items[i]['global_id'])
        for ids in comps.values():
            if len(ids)>1:
                for start in range(0,len(ids),max_size):
                    part=ids[start:start+max_size]
                    if len(part)>1: clusters.append(part)
    return clusters


def merge_entity_records(keep:dict[str,Any],others:list[dict[str,Any]],group:dict[str,Any])->None:
    keep['canonical_name']=(group.get('canonical_name') or keep['canonical_name']).strip(); keep['aliases']=npc.dedupe(keep.get('aliases',[])+group.get('aliases',[])+[e['canonical_name'] for e in others]+sum((e.get('aliases',[]) for e in others),[]),80)
    if group.get('summary'): keep['summary']=group['summary'].strip()
    keep['stable_traits']=npc.dedupe(keep.get('stable_traits',[])+sum((e.get('stable_traits',[]) for e in others),[]),60)
    keep['evidence']=npc.dedupe(keep.get('evidence',[])+sum((e.get('evidence',[]) for e in others),[]),30)
    if not keep.get('visual_global_id'):
        keep['visual_global_id']=next((e.get('visual_global_id','') for e in others if e.get('visual_global_id')), '')
    keep['chapters_seen']=npc.dedupe(keep.get('chapters_seen',[])+sum((e.get('chapters_seen',[]) for e in others),[]),10000)
    for e in others:
        for src in e.get('source_entities',[]):
            if src not in keep['source_entities']: keep['source_entities'].append(src)


def rewrite_ids(graph:dict[str,Any],redirect:dict[str,str])->None:
    def r(x):
        while x in redirect and redirect[x]!=x: x=redirect[x]
        return x
    for f in graph['facts']:
        if f['subject_global_id']: f['subject_global_id']=r(f['subject_global_id'])
        if f['object_global_id']: f['object_global_id']=r(f['object_global_id'])
    for x in graph['relationships']:
        if x['subject_global_id']: x['subject_global_id']=r(x['subject_global_id'])
        if x['object_global_id']: x['object_global_id']=r(x['object_global_id'])
    for e in graph['events']:
        if e['location_global_id']: e['location_global_id']=r(e['location_global_id'])
        for p in e['participants']:
            if p['global_id']: p['global_id']=r(p['global_id'])


def audit_clusters(client,model,registry:list[dict[str,Any]],graph:dict[str,Any],args)->None:
    if args.no_audit:return
    total_merged=0
    for pass_no in range(1,max(1,args.audit_passes)+1):
        clusters=cluster_entities(registry,args.audit_similarity,args.audit_cluster_size)
        print(f'Clustered duplicate audit pass {pass_no}/{max(1,args.audit_passes)}: {len(clusters)} candidate cluster(s)')
        if not clusters: break
        by_id={e['global_id']:e for e in registry}; redirect={}
        for i,ids in enumerate(clusters,start=1):
            npc.progress_start_operation((i-1)/max(1,len(clusters)),1/max(1,len(clusters)))
            data=[compact_entity(by_id[x]) for x in ids if x in by_id]
            if len(data)<2: npc.progress_finish_operation(); continue
            print(f'  auditing cluster {i}/{len(clusters)} ({len(data)} entities)')
            result=npc.chat_json(client,model,AUDIT_SYSTEM,json.dumps(data,ensure_ascii=False,indent=2),AUDIT_SCHEMA,min(args.temperature,.1),max(args.max_tokens,5000))
            for g in result.get('merge_groups',[]):
                keep_id=g.get('keep_global_id'); mids=[x for x in g.get('merge_global_ids',[]) if x in by_id and x!=keep_id]
                if keep_id not in by_id or not mids: continue
                keep=by_id[keep_id]; others=[by_id[x] for x in mids if by_id[x]['entity_type']==keep['entity_type']]
                if not others: continue
                visual_ids={e.get('visual_global_id','') for e in [keep,*others] if e.get('visual_global_id','')}
                if len(visual_ids)>1: continue
                merge_entity_records(keep,others,g)
                for o in others: redirect[o['global_id']]=keep_id; by_id.pop(o['global_id'],None)
            npc.progress_finish_operation()
            if args.delay: time.sleep(args.delay)
        if redirect:
            registry[:]=[e for e in registry if e['global_id'] not in redirect]; rewrite_ids(graph,redirect)
            total_merged+=len(redirect); print(f'  pass {pass_no}: merged {len(redirect)} duplicate global entity ID(s).')
        else:
            print(f'  pass {pass_no}: no merges; stopping audit.')
            break
    print(f'Clustered audit total merged IDs: {total_merged}')


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Consolidate chapter lore into a global knowledge graph.')
    p.add_argument('inputs',nargs='+',type=Path,help='*_lore.json files/directories/* ? wildcard patterns; alphabetical.')
    p.add_argument('--out',type=Path,default=Path('lore_graph.json'))
    p.add_argument('--visual-references',type=Path,default=None,help='Optional consolidated visual registry from Step 2V; aligns character/location/object global IDs.')
    p.add_argument('--candidate-count',type=int,default=12); p.add_argument('--include-all-below',type=int,default=35)
    p.add_argument('--audit-similarity',type=float,default=.68); p.add_argument('--audit-cluster-size',type=int,default=24); p.add_argument('--audit-passes',type=int,default=2); p.add_argument('--no-audit',action='store_true')
    npc.add_common_llm_args(p,max_tokens=7000); p.add_argument('--version',action='version',version=f'%(prog)s {SCRIPT_VERSION}')
    return p


def main()->int:
    args=parser().parse_args(); npc.configure_llm(thinking=args.thinking,chat_backend=args.chat_backend,qwen35_max_output_tokens=args.qwen35_max_output_tokens,qwen35_length_retries=args.qwen35_length_retries)
    paths=npc.discover_inputs(args.inputs,extensions={'.json'}); paths=[p for p in paths if p.name.endswith('_lore.json')]
    if not paths: print('ERROR: no *_lore.json files found.',file=sys.stderr); return 2
    try: chapters=load_chapters(paths); client=npc.make_client(args.base_url,args.api_key); model=npc.select_model(client,args.model)
    except Exception as exc: print(f'ERROR initializing: {exc}',file=sys.stderr); return 1
    npc.init_progress(len(chapters)+(0 if args.no_audit else 1))
    print(f'Script version: {SCRIPT_VERSION}\nModel: {model}\nChapters: {len(chapters)}\n')
    registry=seed_visual_registry(args.visual_references); chapter_mappings={}; graph={'schema_version':OUTPUT_SCHEMA,'script_version':SCRIPT_VERSION,'entities':registry,'facts':[],'relationships':[],'events':[],'terminology':[],'chapter_summaries':{},'unresolved_mentions':[], '_fact_index':{},'_rel_index':{},'_term_index':{}}
    failures=0
    for i,ch in enumerate(chapters,start=1):
        npc.progress_start_item(i); npc.progress_start_operation(0,1)
        try:
            print(f"{ch['chapter_id']}: reconciling {len(ch.get('entities',[]))} entities")
            mapping=reconcile_chapter(client,model,ch,registry,args); chapter_mappings[ch['chapter_id']]=mapping; add_chapter_graph(ch,mapping,registry,graph); npc.progress_finish_operation()
        except Exception as exc: failures+=1; print(f"ERROR {ch.get('chapter_id')}: {exc}",file=sys.stderr)
        npc.progress_advance(1)
    if not args.no_audit:
        npc.progress_start_item(len(chapters)+1); audit_clusters(client,model,registry,graph,args); npc.progress_advance(1)
    for k in ('_fact_index','_rel_index','_term_index'): graph.pop(k,None)
    graph['chapter_order']=[c['chapter_id'] for c in chapters]
    graph['chapter_entity_map']=build_chapter_entity_map(registry)
    graph['source_chapters']=[{'chapter_id':c['chapter_id'],'file':c['_path'],'source_sha256':c.get('source',{}).get('sha256','')} for c in chapters]
    graph['source_digest']=__import__('hashlib').sha256('\n'.join(f"{c['chapter_id']}:{c.get('source',{}).get('sha256','')}" for c in chapters).encode()).hexdigest()
    graph['visual_references_file']=str(args.visual_references) if args.visual_references else None
    graph['indexes']=build_indexes(graph,graph['chapter_order'])
    graph['potential_conflicts']=potential_conflicts(graph['facts'])
    graph['statistics']={k:len(graph[k]) for k in ('entities','facts','relationships','events','terminology','unresolved_mentions')}; graph['statistics']['potential_conflicts']=len(graph['potential_conflicts'])
    args.out.write_text(json.dumps(graph,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"\nSaved {args.out}: {len(registry)} entities, {len(graph['facts'])} facts, {len(graph['relationships'])} relationships, {len(graph['events'])} events, {len(graph['terminology'])} terms")
    return 1 if failures else 0

if __name__=='__main__': raise SystemExit(main())
