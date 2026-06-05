# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 181 |
| Passed | 121 |
| Pass rate | 0.669 |
| Retrieval pass | — |
| Answer pass | 0.669 |
| Agent loop pass | 0.989 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.799 |
| LLM judge count | 179 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 90 | 0.800 | — | 0.800 | 0.989 | — | — | 3.500 |
| worldsense | 91 | 0.538 | — | 0.538 | 0.989 | — | — | 2.090 |

## Failure tags

| tag | count |
| --- | --- |
| `missing_expected_keyword` | 49 |
| `wrong_fact_option` | 46 |
| `missing_citation_kind` | 41 |
| `temporal_order_error` | 14 |
| `wrong_temporal_option` | 14 |
| `missing_frame_or_slide` | 13 |
| `no_final_answer` | 7 |
| `missing_slide` | 1 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `vme-334-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, temporal_order_error, wrong_temporal_option; selected=-; recommended=B | What is the correct order in which the following patterns appear in the video?  Candidates: A) Pizza parlors, the Uni... |
| `vme-354-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=C | How does the speaker introduce the fact that the Mediterranean might disappear for a long while?  Candidates: A) MSC.... |
| `vme-381-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=D | How many clips of Mad Men where the typewriter made appearances are shown in the video?  Candidates: A) 3. B) 5. C) 6... |
| `vme-394-3` | answer | missing_citation_kind, missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=D; recommended=B | In the video, in the little girl's recollection, what is the correct order of the events she experienced? (1) Her pai... |
| `vme-400-2` | answer | missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | How does Puss in Boots feel when he hear the whistle of the man with hat?  Candidates: A) Anxious. B) Exited. C) Fear... |
| `vme-426-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, temporal_order_error, wrong_temporal_option; selected=-; recommended=A | When does the interview with the woman in the green shirt take place in the video?  Candidates: A) Middle of the vide... |
| `vme-455-1` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=D | What are the players doing before the match formally begins, according to the video?  Candidates: A) Exchanging team ... |
| `vme-455-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=A | In the match, which player commits the first foul?  Candidates: A) Thailand player number 21. B) Thailand player numb... |
| `vme-458-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=C | Which player is injured in the middle of the game?  Candidates: A) Korea's player number 12. B) Malaysia's player num... |
| `vme-496-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, temporal_order_error, wrong_temporal_option; selected=-; recommended=D | According to this video, in which order do the following events happen? (a) The magician took the white cloth away fr... |
