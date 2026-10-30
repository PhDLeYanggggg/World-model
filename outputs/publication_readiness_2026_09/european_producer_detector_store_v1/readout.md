# Complete Producer Past-Input Coverage

## Material Passport

Fresh_run: full coverage reconciliation,new raw input extraction and separate
complete raw reparse. Cached_verified: old controller input blocks,original
SOURCE roles/packed identities. New trajectory training/scoring is not_run in
this input receipt. No independent or controller-held readout is opened.

The original producer uses318,969queries from163SOURCE-training recordings.
The previous controller cache contained149,510queries/74recordings,leaving
169,459queries across89recordings. All original producer rows are now present;
there is no zero padding,query dropping or target/split change.

| Execution | PID | Seconds | Peak RSS bytes | Fresh raw records | Old verified blocks | Pilot checkpoint reused |
|---|---:|---:|---:|---:|---:|---:|
| Full build | 46919 | 45.337349 | 1,725,988,864 | 88 | 74 | 1 |
| Separate complete raw reparse | 47116 | 78.277357 | 2,616,459,264 | 163 | 0 | 0 |

Both exit0. All318,969input arrays and the149,510old overlapping rows agree
exactly. There are489actual first/middle/last raw-prefix checks across the
complete reparse. Every lookup preserves original eight boxes,confidence,
midpoint history,current width,and recording/agent/frame IDs. Final NPY
arrays total111,001,980bytes and remain private with recovery blocks.

The64-field schema is unchanged. Normalizers are not fitted by this store;
every later producer must use only its own fitting localities. No future
target array,future support mask or goal label enters extraction. Producer
training on original own-fit localities and evaluation of controller-held
recordings are distinct permissions; this store does not open the latter.

The new Transformer branch has only synthetic code tests so far,not a real
SOURCE training result. That subsequent study must first pass a real training
and exact-recovery pilot,then retain all18original designs by four matched
input arms,with full4,000updates/model. The current controller safety failure
is not reclassified by successful input alignment.

SOURCE remains supplied detector-silver midpoint pixels,obs8/pred12/stride12
raw frames. Sensor-as-of,scale,homography and effective timing are not
certified. No human-gold,true3D,foundation,deployment,Stage5C or SMC claim.
