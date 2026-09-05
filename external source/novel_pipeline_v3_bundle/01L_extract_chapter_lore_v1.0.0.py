#!/usr/bin/env python3
"""Extract chapter-level lore/knowledge-graph candidates from very large novels.

Outputs one *_lore.json per chapter. Designed for arbitrarily large corpora by using
small source chunks and hierarchical LLM merges instead of one monolithic chapter merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import novel_pipeline_common as npc

SCRIPT_VERSION='1.0.0'
SCHEMA_VERSION='novel-lore.chapter.v1'
ENTITY_TYPES=['character','location','object','artifact','organization','people','species','language','concept','custom','other']
IMPORTANCE=['major','recurring','minor','background']

ENTITY_SCHEMA={
    'type':'object','properties':{
        'canonical_name':{'type':'string','maxLength':120},
        'aliases':{'type':'array','maxItems':10,'items':{'type':'string','maxLength':100}},
        'entity_type':{'type':'string','enum':ENTITY_TYPES},
        'summary':{'type':'string','maxLength':700},
        'stable_traits':{'type':'array','maxItems':10,'items':{'type':'string','maxLength':180}},
        'importance':{'type':'string','enum':IMPORTANCE},
        'evidence':{'type':'array','maxItems':4,'items':{'type':'string','maxLength':160}},
    },'required':['canonical_name','aliases','entity_type','summary','stable_traits','importance','evidence'],'additionalProperties':False
}
FACT_SCHEMA={
    'type':'object','properties':{
        'subject_name':{'type':'string','maxLength':120},
        'predicate':{'type':'string','maxLength':100},
        'object_name':{'type':'string','maxLength':120},
        'object_text':{'type':'string','maxLength':300},
        'qualifiers':{'type':'array','maxItems':8,'items':{'type':'string','maxLength':140}},
        'temporal_scope':{'type':'string','maxLength':180},
        'confidence':{'type':'number','minimum':0,'maximum':1},
        'evidence':{'type':'array','maxItems':3,'items':{'type':'string','maxLength':160}},
    },'required':['subject_name','predicate','object_name','object_text','qualifiers','temporal_scope','confidence','evidence'],'additionalProperties':False
}
REL_SCHEMA={
    'type':'object','properties':{
        'subject_name':{'type':'string','maxLength':120},
        'relation':{'type':'string','maxLength':100},
        'object_name':{'type':'string','maxLength':120},
        'state':{'type':'string','maxLength':220},
        'confidence':{'type':'number','minimum':0,'maximum':1},
        'evidence':{'type':'array','maxItems':3,'items':{'type':'string','maxLength':160}},
    },'required':['subject_name','relation','object_name','state','confidence','evidence'],'additionalProperties':False
}
EVENT_SCHEMA={
    'type':'object','properties':{
        'title':{'type':'string','maxLength':140},
        'summary':{'type':'string','maxLength':700},
        'participants':{'type':'array','maxItems':16,'items':{'type':'string','maxLength':120}},
        'location_name':{'type':'string','maxLength':120},
        'sequence_hint':{'type':'string','maxLength':180},
        'causes':{'type':'array','maxItems':6,'items':{'type':'string','maxLength':180}},
        'consequences':{'type':'array','maxItems':6,'items':{'type':'string','maxLength':180}},
        'evidence':{'type':'array','maxItems':4,'items':{'type':'string','maxLength':160}},
    },'required':['title','summary','participants','location_name','sequence_hint','causes','consequences','evidence'],'additionalProperties':False
}
TERM_SCHEMA={
    'type':'object','properties':{
        'term':{'type':'string','maxLength':120},
        'definition':{'type':'string','maxLength':500},
        'category':{'type':'string','maxLength':80},
        'aliases':{'type':'array','maxItems':8,'items':{'type':'string','maxLength':100}},
        'evidence':{'type':'array','maxItems':3,'items':{'type':'string','maxLength':160}},
    },'required':['term','definition','category','aliases','evidence'],'additionalProperties':False
}
LORE_SCHEMA={
    'name':'chapter_lore_v1','strict':True,'schema':{
        'type':'object','properties':{
            'chapter_summary':{'type':'string','maxLength':1200},
            'entities':{'type':'array','items':ENTITY_SCHEMA},
            'facts':{'type':'array','items':FACT_SCHEMA},
            'relationships':{'type':'array','items':REL_SCHEMA},
            'events':{'type':'array','items':EVENT_SCHEMA},
            'terminology':{'type':'array','items':TERM_SCHEMA},
        },'required':['chapter_summary','entities','facts','relationships','events','terminology'],'additionalProperties':False
    }
}

EXTRACT_SYSTEM=r'''
Extract a compact, source-grounded lore graph from the supplied novel passage.

Capture information useful for long-range continuity, worldbuilding, retrieval, and
later scene generation. Do not limit yourself to visual details.

ENTITIES may include characters, locations, objects/artifacts, organizations, peoples,
species, languages, concepts, customs, and other named or clearly recurring things.
FACTS are atomic propositions. Prefer normalized predicates such as born_in, member_of,
located_in, owns, carries, knows, speaks_language, title, role, origin, property,
believes, rule, custom, ability, status, or another concise snake_case relation.
RELATIONSHIPS represent durable or narratively important entity-to-entity relations.
EVENTS represent actual happenings in this passage, with causes/consequences only when
supported. TERMINOLOGY records explicit setting-specific terms and definitions.

Grounding rules:
- use only the supplied passage;
- never infer hidden lore merely because it is famous or likely;
- preserve uncertainty with confidence < 1.0;
- evidence is a SHORT anchor, not a long quotation;
- do not reproduce copyrighted prose unnecessarily;
- distinguish persistent facts from temporary states using temporal_scope/state;
- object_name is for a named entity; otherwise leave it empty and use object_text;
- merge obvious aliases within the passage, but do not guess that two ambiguous names
  are the same entity;
- keep the JSON compact enough to merge hierarchically later.
'''.strip()

MERGE_SYSTEM=r'''
Merge several partial lore graphs from adjacent/overlapping portions of ONE chapter.
Deduplicate only clearly equivalent entities/facts/relationships/events/terms. Preserve
all distinct source-supported information, aliases, uncertainty, temporal distinctions,
and evidence anchors. Never collapse two different entities merely because names or
roles are similar. Keep facts atomic and normalize predicates. The output must remain
compact: union evidence selectively instead of repeating near-identical anchors.
'''.strip()


def _clean(data: dict[str,Any]) -> dict[str,Any]:
    out={
        'chapter_summary':str(data.get('chapter_summary','')).strip(),
        'entities':[], 'facts':[], 'relationships':[], 'events':[], 'terminology':[]
    }
    for e in data.get('entities',[]):
        name=str(e.get('canonical_name','')).strip()
        if not name: continue
        out['entities'].append({
            'canonical_name':name,'aliases':npc.dedupe(e.get('aliases',[]),10),
            'entity_type':e.get('entity_type','other') if e.get('entity_type') in ENTITY_TYPES else 'other',
            'summary':str(e.get('summary','')).strip(),'stable_traits':npc.dedupe(e.get('stable_traits',[]),10),
            'importance':e.get('importance','minor') if e.get('importance') in IMPORTANCE else 'minor',
            'evidence':npc.dedupe(e.get('evidence',[]),4),
        })
    for key in ('facts','relationships','events','terminology'):
        for item in data.get(key,[]):
            if isinstance(item,dict): out[key].append(item)
    return out


def _merge_call(client,model,chapter_id:str,parts:list[dict[str,Any]],args,round_no:int,batch_no:int) -> dict[str,Any]:
    user=f"Chapter: {chapter_id}\nMerge round {round_no}, batch {batch_no}\n\nPARTIAL LORE GRAPHS:\n{json.dumps(parts,ensure_ascii=False,indent=2)}"
    return _clean(npc.chat_json(client,model,MERGE_SYSTEM,user,LORE_SCHEMA,min(args.temperature,.14),max(args.max_tokens,7000)))


def _merge_operation_count(n:int,batch:int) -> int:
    count=0
    while n>1:
        groups=math.ceil(n/batch); count+=groups; n=groups
    return count


def hierarchical_merge(client,model,chapter_id:str,parts:list[dict[str,Any]],args,cache_dir:Path,op_index:int,op_span:float) -> tuple[dict[str,Any],int]:
    if len(parts)==1: return _clean(parts[0]),op_index
    level=[_clean(x) for x in parts]; round_no=1
    while len(level)>1:
        next_level=[]
        for batch_no,start in enumerate(range(0,len(level),args.merge_batch_size),start=1):
            batch=level[start:start+args.merge_batch_size]
            npc.progress_start_operation(op_index*op_span,op_span)
            key=hashlib.sha256((SCHEMA_VERSION+'\n'+model+'\n'+json.dumps(batch,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()
            cp=cache_dir/f'merge_r{round_no:02d}_b{batch_no:03d}.json'; merged=None
            if cp.exists() and not args.force:
                try:
                    c=json.loads(cp.read_text(encoding='utf-8'))
                    if c.get('cache_key')==key: merged=_clean(c['result'])
                except Exception: pass
            if merged is None:
                print(f'  hierarchical merge round {round_no}, batch {batch_no}/{math.ceil(len(level)/args.merge_batch_size)} ({len(batch)} partial graphs)')
                merged=_merge_call(client,model,chapter_id,batch,args,round_no,batch_no)
                cp.write_text(json.dumps({'cache_key':key,'result':merged},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                npc.progress_finish_operation()
                if args.delay: time.sleep(args.delay)
            else:
                npc.progress_advance((op_index+1)*op_span); print(f'  merge r{round_no} b{batch_no}: cached')
            op_index+=1; next_level.append(merged)
        level=next_level; round_no+=1
    return level[0],op_index


def assign_local_ids(data:dict[str,Any]) -> dict[str,Any]:
    counters={}; entities=[]
    prefix={'character':'CHAR','location':'LOC','object':'OBJ','artifact':'ART','organization':'ORG','people':'PEO','species':'SPEC','language':'LANG','concept':'CON','custom':'CUS','other':'ENT'}
    for e in data['entities']:
        typ=e['entity_type']; counters[typ]=counters.get(typ,0)+1
        entities.append({'local_id':f"{prefix[typ]}_{counters[typ]:03d}",**e})
    return {**data,'entities':entities}


def process(path:Path,out_dir:Path,client,model,args) -> Path:
    chapter_id=npc.slug(path.stem); text=npc.read_text_document(path)
    chunks=npc.split_chunks(text,max(3000,args.chunk_chars),max(0,args.overlap_paragraphs))
    cache=out_dir/'.cache'/chapter_id; cache.mkdir(parents=True,exist_ok=True)
    merge_ops=_merge_operation_count(len(chunks),args.merge_batch_size); total_ops=max(1,len(chunks)+merge_ops); span=1/total_ops; op=0
    print(f'{path.name}: {len(text):,} chars, {len(chunks)} chunk(s), {merge_ops} hierarchical merge call(s)')
    partial=[]
    for i,chunk in enumerate(chunks,start=1):
        npc.progress_start_operation(op*span,span)
        key=hashlib.sha256((SCHEMA_VERSION+'\n'+model+'\n'+chunk).encode()).hexdigest(); cp=cache/f'chunk_{i:04d}.json'; result=None
        if cp.exists() and not args.force:
            try:
                c=json.loads(cp.read_text(encoding='utf-8'))
                if c.get('cache_key')==key: result=_clean(c['result'])
            except Exception: pass
        if result is None:
            print(f'  extracting lore chunk {i}/{len(chunks)}')
            user=f'Chapter: {chapter_id}\nPassage chunk: {i}/{len(chunks)}\n\n--- BEGIN NOVEL PASSAGE ---\n{chunk}\n--- END NOVEL PASSAGE ---'
            result=_clean(npc.chat_json(client,model,EXTRACT_SYSTEM,user,LORE_SCHEMA,args.temperature,args.max_tokens))
            cp.write_text(json.dumps({'cache_key':key,'result':result},ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); npc.progress_finish_operation()
            if args.delay: time.sleep(args.delay)
        else:
            npc.progress_advance((op+1)*span); print(f'  lore chunk {i}/{len(chunks)}: cached')
        op+=1; partial.append(result)
    merged,op=hierarchical_merge(client,model,chapter_id,partial,args,cache,op,span)
    npc.progress_advance(1.0)
    merged=assign_local_ids(merged)
    payload={'schema_version':SCHEMA_VERSION,'script_version':SCRIPT_VERSION,'chapter_id':chapter_id,
             'source':{'file':path.name,'absolute_path':str(path.resolve()),'character_count':len(text)},
             'llm':{'base_url':args.base_url,'model':model,'thinking':args.thinking,'chat_backend':args.chat_backend},**merged}
    out=out_dir/f'{chapter_id}_lore.json'; out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"  saved {out.name}: {len(payload['entities'])} entities, {len(payload['facts'])} facts, {len(payload['relationships'])} relationships, {len(payload['events'])} events, {len(payload['terminology'])} terms")
    return out


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Extract scalable chapter-level lore graphs from novels.')
    p.add_argument('inputs',nargs='+',type=Path,help='Chapter files/directories/* ? wildcard patterns; processed alphabetically.')
    p.add_argument('--out-dir',type=Path,default=Path('chapter_lore'))
    p.add_argument('--chunk-chars',type=int,default=7000)
    p.add_argument('--overlap-paragraphs',type=int,default=2)
    p.add_argument('--merge-batch-size',type=int,default=6,help='Maximum partial lore graphs per hierarchical merge call.')
    npc.add_common_llm_args(p,max_tokens=6500)
    p.add_argument('--version',action='version',version=f'%(prog)s {SCRIPT_VERSION}')
    return p


def main()->int:
    args=parser().parse_args(); args.merge_batch_size=max(2,args.merge_batch_size)
    npc.configure_llm(thinking=args.thinking,chat_backend=args.chat_backend,qwen35_max_output_tokens=args.qwen35_max_output_tokens,qwen35_length_retries=args.qwen35_length_retries)
    files=npc.discover_inputs(args.inputs)
    if not files: print('ERROR: no supported chapter files found.',file=sys.stderr); return 2
    args.out_dir.mkdir(parents=True,exist_ok=True); npc.init_progress(len(files))
    try:
        client=npc.make_client(args.base_url,args.api_key); model=npc.select_model(client,args.model)
    except Exception as exc: print(f'ERROR connecting to LM Studio: {exc}',file=sys.stderr); return 1
    print(f'Script version: {SCRIPT_VERSION}\nLM Studio: {args.base_url}\nModel: {model}\nThinking: {"enabled" if args.thinking else "disabled"}\nInputs: {len(files)}\n')
    failures=0
    for idx,path in enumerate(files,start=1):
        npc.progress_start_item(idx)
        try: process(path,args.out_dir,client,model,args)
        except Exception as exc: failures+=1; print(f'ERROR {path}: {exc}',file=sys.stderr)
        finally: npc.progress_advance(1.0)
    print(f'\nCompleted: {len(files)-failures}/{len(files)} chapter(s).')
    return 1 if failures else 0

if __name__=='__main__': raise SystemExit(main())
