# Masked Zara Past-Image Input Store

Fresh diagnostic input construction; no new training, prediction result or formal cohort admission.

| Recording | Source rows | Past8 windows | Full / partial / empty crops | Every frame partly observed | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| ucy_zara01 | 5024 | 3988 | 4403 / 621 / 0 | 3988 | 21033632 |
| ucy_zara02 | 9537 | 8110 | 8196 / 1341 / 0 | 8110 | 39944129 |

Pixel coverage is retained per block. No recentering, hallucinated pixels or silent row removal.
Each crop is stored once per source row; past windows are index lists, not materialized eight times.
Explicit offline diagnostic mode preserves later-control provenance. Strict control-as-of mode rejects affected queries.
Formal supervised/evaluation roles are rejected pending the scientific observation decision.
