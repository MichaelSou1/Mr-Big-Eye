# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 181 |
| Passed | 99 |
| Pass rate | 0.547 |
| Retrieval pass | — |
| Answer pass | 0.547 |
| Agent loop pass | 0.994 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.247 |
| LLM judge count | 178 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 90 | 0.700 | — | 0.700 | 0.989 | — | — | 2.811 |
| worldsense | 91 | 0.396 | — | 0.396 | 1.000 | — | — | 1.670 |

## Failure tags

| tag | count |
| --- | --- |
| `missing_citation_kind` | 72 |
| `wrong_fact_option` | 60 |
| `missing_expected_keyword` | 49 |
| `temporal_order_error` | 22 |
| `wrong_temporal_option` | 22 |
| `missing_frame_or_slide` | 16 |
| `missing_slide` | 3 |
| `agent_loop_issue` | 1 |
| `no_final_answer` | 1 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `vme-324-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=D | Why should exercise be included as a part of maintaining a healthy lymphatic system, according to the video?  Candida... |
| `vme-340-1` | answer | missing_citation_kind, wrong_fact_option; selected=C; recommended=A | What is the intended educational purpose of this video for the viewer?  Candidates: A) How the stock market works. B)... |
| `vme-354-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=D; recommended=C | How does the speaker introduce the fact that the Mediterranean might disappear for a long while?  Candidates: A) MSC.... |
| `vme-382-2` | answer | missing_citation_kind, wrong_fact_option; selected=C; recommended=D | According to the video, which statement is correct?  Candidates: A) IMAX cameras use 35mm film. B) 1000 feet of film ... |
| `vme-382-3` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=D | Which IMAX movie isn't in the video?  Candidates: A) The Hunger Games: Catching Fire. B) The Dark Knight. C) Oppenhei... |
| `vme-394-3` | answer | missing_citation_kind, temporal_order_error, wrong_temporal_option; selected=D; recommended=B | In the video, in the little girl's recollection, what is the correct order of the events she experienced? (1) Her pai... |
| `vme-400-3` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=A | How many crystals with cat in boots does Death break?  Candidates: A) 8. B) 6. C) 7. D) 9. |
| `vme-455-1` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=D | What are the players doing before the match formally begins, according to the video?  Candidates: A) Exchanging team ... |
| `vme-496-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, temporal_order_error, wrong_temporal_option; selected=-; recommended=D | According to this video, in which order do the following events happen? (a) The magician took the white cloth away fr... |
| `vme-507-2` | answer | missing_citation_kind, temporal_order_error, wrong_temporal_option; selected=B; recommended=A | Based on the order of introduction in the video, which of the following options is introduced last?  Candidates: A) L... |
