# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 60 |
| Passed | 36 |
| Pass rate | 0.600 |
| Retrieval pass | — |
| Answer pass | 0.600 |
| Agent loop pass | 1.000 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.621 |
| LLM judge count | 58 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 30 | 0.667 | — | 0.667 | 1.000 | — | — | 3.167 |
| worldsense | 30 | 0.533 | — | 0.533 | 1.000 | — | — | 2.036 |

## Failure tags

| tag | count |
| --- | --- |
| `missing_citation_kind` | 20 |
| `wrong_fact_option` | 16 |
| `missing_expected_keyword` | 14 |
| `temporal_order_error` | 8 |
| `wrong_temporal_option` | 8 |
| `missing_frame_or_slide` | 6 |
| `no_final_answer` | 4 |
| `missing_slide` | 1 |
| `post_answer_retrieval` | 1 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `vme-313-2` | answer | missing_citation_kind, wrong_fact_option; selected=C; recommended=B | What is the teacher's attitude toward the student when the student attempts to play middle C on the violin?  Candidat... |
| `vme-400-2` | answer | missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | How does Puss in Boots feel when he hear the whistle of the man with hat?  Candidates: A) Anxious. B) Exited. C) Fear... |
| `vme-426-2` | answer | missing_citation_kind, post_answer_retrieval, temporal_order_error, wrong_temporal_option; selected=B; recommended=A | When does the interview with the woman in the green shirt take place in the video?  Candidates: A) Middle of the vide... |
| `vme-455-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=A | In the match, which player commits the first foul?  Candidates: A) Thailand player number 21. B) Thailand player numb... |
| `vme-496-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=B | What is the last magic the magician played?  Candidates: A) He produced a small leather. B) He produced a large leath... |
| `vme-510-2` | answer | missing_citation_kind, temporal_order_error, wrong_temporal_option; selected=D; recommended=C | In which part of the video does the red parrot appear?  Candidates: A) The parrot does not appear. B) End of the vide... |
| `vme-517-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=-; recommended=D | What happened after the two actresses climbed onto the high board?  Candidates: A) Two actresses jumped to the swing.... |
| `vme-544-2` | answer | missing_citation_kind, temporal_order_error, wrong_temporal_option; selected=D; recommended=B | What is the correct order in which the characters appear in the video?  Candidates: A) Dog walkers, motorcycle riders... |
| `vme-555-2` | answer | missing_citation_kind, missing_slide, wrong_fact_option; selected=B; recommended=D | What time does the hero in the video get up in real life?  Candidates: A) 11:00 AM. B) 7:00 AM. C) 10:30 AM. D) 10:00... |
| `vme-558-2` | answer | missing_citation_kind, missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=C; recommended=B | Which of the following options correctly orders the sequence of the boy's daily itinerary?  Candidates: A) Go to clas... |
