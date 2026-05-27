# User Action Required: Stage42-DS Source Conversion Readiness

本步骤发现了若干本地路径和 derived cache，但没有把它们当作 conversion-ready。继续转换前需要用户确认 legal/source 信息。

| dataset | local/path status | required action |
| --- | --- | --- |
| `ucy_crowd_original` | raw path found, derived cache found | user must confirm terms acceptance, allowed use, acceptance date, local path, and source identity before conversion |
| `eth_biwi_original` | raw path found, derived cache found | user must confirm terms acceptance, allowed use, acceptance date, local path, and source identity before conversion |
| `trajnetplusplus_official` | raw path found, derived cache found | user must confirm terms acceptance, allowed use, acceptance date, local path, and source identity before conversion |
| `opentraj_toolkit` | raw path found, derived cache found | user must confirm terms acceptance, allowed use, acceptance date, local path, and source identity before conversion |
| `aerialmpt_or_other_topdown` | raw path found, derived cache found | provide/verify official dataset URL before any conversion claim |
| `stanford_drone_dataset` | raw path found, derived cache found | keep_as_sdd_pixel_raw_frame_reference; do not count as new external source |
| `tgsim_diagnostic` | missing | keep_as_diagnostic_only; do not use as pedestrian topdown official benchmark |

必须提供的信息：

- official dataset/source URL
- 是否已接受条款、接受日期、允许用途
- 本地 raw path
- source identity / dataset version
- 是否允许 derived conversion / redistribution / publication use

在这些信息缺失前，任何 local path 或 derived cache 都不能写成 legally converted dataset。
