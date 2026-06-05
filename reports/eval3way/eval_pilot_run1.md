# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 181 |
| Passed | 114 |
| Pass rate | 0.630 |
| Retrieval pass | — |
| Answer pass | 0.630 |
| Agent loop pass | 1.000 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.494 |
| LLM judge count | 178 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 90 | 0.722 | — | 0.722 | 1.000 | — | — | 3.135 |
| worldsense | 91 | 0.538 | — | 0.538 | 1.000 | — | — | 1.854 |

## Failure tags

| tag | count |
| --- | --- |
| `missing_expected_keyword` | 55 |
| `wrong_fact_option` | 53 |
| `missing_citation_kind` | 45 |
| `temporal_order_error` | 14 |
| `wrong_temporal_option` | 14 |
| `missing_frame_or_slide` | 13 |
| `no_final_answer` | 8 |
| `missing_slide` | 2 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `vme-354-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=-; recommended=C | How does the speaker introduce the fact that the Mediterranean might disappear for a long while?  Candidates: A) MSC.... |
| `vme-381-3` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=B; recommended=B | Which year marked the debut of the typewriter introduced in the video?  Candidates: A) 1961. B) 1971. C) 1981. D) 1986. |
| `vme-382-3` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=D | Which IMAX movie isn't in the video?  Candidates: A) The Hunger Games: Catching Fire. B) The Dark Knight. C) Oppenhei... |
| `vme-394-3` | answer | missing_citation_kind, missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=D; recommended=B | In the video, in the little girl's recollection, what is the correct order of the events she experienced? (1) Her pai... |
| `vme-400-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=-; recommended=B | Where is Puss in Boots meet Death?  Candidates: A) A kitchen. B) An office. C) A hole. D) A bar. |
| `vme-400-2` | answer | missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | How does Puss in Boots feel when he hear the whistle of the man with hat?  Candidates: A) Anxious. B) Exited. C) Fear... |
| `vme-423-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=D | In the video, the man speaking to the camera, wearing glasses and a brown coat, is most likely in which role?  Candid... |
| `vme-455-1` | answer | missing_expected_keyword, wrong_fact_option; selected=B; recommended=D | What are the players doing before the match formally begins, according to the video?  Candidates: A) Exchanging team ... |
| `vme-458-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=C | Which player is injured in the middle of the game?  Candidates: A) Korea's player number 12. B) Malaysia's player num... |
| `vme-496-1` | answer | missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=B; recommended=D | According to this video, in which order do the following events happen? (a) The magician took the white cloth away fr... |
