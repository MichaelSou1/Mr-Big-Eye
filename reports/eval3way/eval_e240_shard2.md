# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 60 |
| Passed | 44 |
| Pass rate | 0.733 |
| Retrieval pass | — |
| Answer pass | 0.733 |
| Agent loop pass | 0.983 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 3.305 |
| LLM judge count | 59 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 30 | 0.767 | — | 0.767 | 1.000 | — | — | 3.333 |
| worldsense | 30 | 0.700 | — | 0.700 | 0.967 | — | — | 3.276 |

## Failure tags

| tag | count |
| --- | --- |
| `missing_expected_keyword` | 13 |
| `missing_citation_kind` | 12 |
| `wrong_fact_option` | 12 |
| `missing_frame_or_slide` | 7 |
| `no_final_answer` | 5 |
| `temporal_order_error` | 4 |
| `wrong_temporal_option` | 4 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `vme-334-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, temporal_order_error, wrong_temporal_option; selected=-; recommended=B | What is the correct order in which the following patterns appear in the video?  Candidates: A) Pizza parlors, the Uni... |
| `vme-354-3` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=C; recommended=D | What is Nuralagus rex doing in the video?  Candidates: A) Flying. B) Eating. C) Walking. D) Hopping. |
| `vme-382-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=D | Which IMAX movie isn't in the video?  Candidates: A) The Hunger Games: Catching Fire. B) The Dark Knight. C) Oppenhei... |
| `vme-394-3` | answer | missing_citation_kind, missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=D; recommended=B | In the video, in the little girl's recollection, what is the correct order of the events she experienced? (1) Her pai... |
| `vme-496-3` | answer | missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | What sentence best describes the performance?  Candidates: A) The magic is not wonderful enough to make the audiences... |
| `vme-568-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=-; recommended=C | After telling the story about the railroad, where does the main character in the video reach?  Candidates: A) Los Ang... |
| `vme-588-3` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, wrong_fact_option; selected=-; recommended=C | What is the highest weight that the boys' group can successfully bench press in the video?  Candidates: A) 405 lbs. B... |
| `worldsense-SbbeAnCD-task0` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=C; recommended=B | Is there a change in the background sound in the video?  Candidates: A) I am not sure. B) Yes. C) No. |
| `worldsense-UYkFSXsh-task1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, temporal_order_error, wrong_temporal_option; selected=-; recommended=D | At what point in the video do the audience members cheer?  Candidates: A) In the middle of the video. B) Throughout t... |
| `worldsense-WybMuwJr-task0` | answer | missing_citation_kind, wrong_fact_option; selected=A; recommended=C | How many times does the whistle sound appear in the video?  Candidates: A) One. B) Six. C) Four. D) Two. |
