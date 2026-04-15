from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_long_memory.evaluation.dataset_loader import iter_history_messages, load_stream
from llm_long_memory.memory.memory_manager import MemoryManager
from llm_long_memory.utils.helpers import load_config, resolve_project_path


class DryRunLLM:
    model_name = 'dry-run-no-final-8b'

    def chat(self, messages):
        raise RuntimeError('Final 8B should not be called in dry-run')


cfg = load_config()
cfg = copy.deepcopy(cfg)
root_tmp = Path(tempfile.mkdtemp(prefix='ragdebug_multidry_'))
mid_db = root_tmp / 'mid.db'
long_db = root_tmp / 'long.db'
cfg['memory']['mid_memory']['database_file'] = str(mid_db)
cfg['memory']['long_memory']['database_file'] = str(long_db)
cfg['memory']['long_memory']['extractor']['model'] = 'qwen3:8b'
cfg['memory']['long_memory']['extractor']['timeout_sec'] = 180
cfg['memory']['long_memory']['extractor']['max_output_tokens'] = 220
cfg['memory']['long_memory']['extractor']['retry_max_attempts'] = 1
cfg['memory']['long_memory']['query_struct']['enabled'] = False
cfg['retrieval']['answering']['second_pass_llm_enabled'] = False
cfg['retrieval']['answering']['llm_fallback_to_top_candidate'] = False
cfg['retrieval']['answering']['graph_refiner_enabled'] = True
cfg['retrieval']['answering']['graph_context_from_store_enabled'] = True
cfg['memory']['long_memory']['enabled'] = True
cfg['memory']['long_memory']['offline_graph']['enabled'] = True

instances = list(load_stream(str(resolve_project_path('data/raw/LongMemEval/longmemeval_ragdebug10_rebuilt.json'))))
seen_types = []
selected = []
for inst in instances:
    qtype = str(inst.get('question_type', ''))
    if qtype not in seen_types:
        seen_types.append(qtype)
        selected.append(inst)
    if len(selected) >= 6:
        break

print('TMPDIR=', root_tmp)
print('SELECTED:')
for i, inst in enumerate(selected, 1):
    print(f"{i}. {inst.get('question_id')} | {inst.get('question_type')} | {inst.get('question')}")

manager = MemoryManager(llm=DryRunLLM(), config=cfg)
try:
    for idx, inst in enumerate(selected, 1):
        qid = str(inst.get('question_id', ''))
        qtype = str(inst.get('question_type', ''))
        question = str(inst.get('question', '')).strip()
        print('\n' + '=' * 120)
        print(f'SAMPLE {idx} | qid={qid} | type={qtype}')
        print('QUESTION:', question)
        manager.reset_for_new_instance()
        for message in iter_history_messages(inst):
            manager.ingest_message(message)
        manager.finalize_ingest()
        manager.archive_short_to_mid(clear_short=True)
        precomputed_context = manager.retrieve_context(question)
        _ctx, topics, chunks = precomputed_context
        accepted = manager.offline_build_long_graph_from_chunks(chunks, query=question)
        topics2, chunks2, evidence_sentences, candidates, fallback_answer, evidence_candidate, best_evidence, best_candidate = manager._prepare_answer_inputs(question, precomputed_context)
        prompt_text = manager._build_generation_prompt(
            input_text=question,
            chunks=chunks2,
            candidates=candidates,
            best_evidence=best_evidence,
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
        )
        sections = manager.get_last_prompt_eval_chunks()
        stats = manager.long_memory.debug_stats()
        print('retrieved_topics=', len(topics2), 'retrieved_chunks=', len(chunks2), 'accepted_events=', accepted)
        print('evidence_sentences=', len(evidence_sentences), 'candidates=', len(candidates), 'best_evidence=', repr(best_evidence[:140]))
        print('fallback_answer=', repr(fallback_answer), 'evidence_candidate=', repr((evidence_candidate or {}).get('answer', '')), 'best_candidate=', repr(best_candidate))
        print('graph_stats: events=', stats.get('events', 0), 'accepted=', stats.get('ingest_event_accepted', 0), 'rejected=', stats.get('ingest_event_rejected', 0))
        print('prompt_chars=', len(prompt_text), 'sections=', len(sections))
        for sec in sections:
            text = str(sec.get('text', '')).strip()
            print('--- SECTION:', sec.get('section', ''), '| chars=', len(text))
            if text:
                preview = text.replace('\n', '\\n')
                if len(preview) > 700:
                    preview = preview[:700] + '...'
                print(preview)
            else:
                print('(empty)')
finally:
    manager.close()
