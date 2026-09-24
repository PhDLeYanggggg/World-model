# HT21 数据审计与复核

本轮完成的是实际数据获取、逐行结构审计和输入边界验证，不是训练或预测性能实验。
标注文件保存在本地 Git 忽略目录，不随仓库分发。

## 已运行

在项目根目录使用 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/fetch_m3w_ht21_annotations.py --download
.venv-pytorch/bin/python scripts/audit_m3w_ht21_annotations.py
.venv-pytorch/bin/python scripts/audit_m3w_ht21_annotations.py --verify
.venv-pytorch/bin/python scripts/audit_m3w_ht21_cadence.py
.venv-pytorch/bin/python scripts/audit_m3w_ht21_cadence.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_ht21_support.py
.venv-pytorch/bin/python scripts/verify_m3w_ht21_boundary.py
.venv-pytorch/bin/python scripts/verify_m3w_ht21_boundary.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_ht21_intake.py tests/test_m3w_ht21_cadence.py tests/test_m3w_ht21_admission.py -q
```

结果：36 项针对性测试通过；全部 1,188,496 行复核一致；独立 CSV/set
计数通过 28 组检查；96 个真实历史窗口在删除/改写未来行、合并静止标签后
保持相同输入。独立计数指不同算法实现，不是另一个研究团队的重复实验。

原始解析约 2.71 秒，该数字不包括下载、其他检查和本轮总耗时。
没有训练 PID、GPU 作业或后台进程需要等待。CREATE 当前连接仍有已记录的
认证问题；没有为了本次轻量审计重试凭据或提交远程任务。

## 再次复核

已有报告采用不覆盖写入。上面的首次执行命令不能不加区分地全部重跑，
尤其首次审计及其首次 `--verify` 会保留不可覆盖的回执。
不要删除旧结果来制造新的 fresh_run。可重复执行的只读复核是：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_ht21_cadence.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_ht21_boundary.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_ht21_intake.py tests/test_m3w_ht21_cadence.py tests/test_m3w_ht21_admission.py -q
```

两个复核都会读取并核对原始归档哈希，且重新读取实际 GT。
`verification.json` 保存完整审计的首次精确回放；
`independent_verification.json` 保存全部 GT 的独立算法计数回执。
新增算法或协议需要新版本输出，不得覆盖当前事实记录。

## 不能据此宣称的事

- 118.85 万行不是 118.85 万独立样本，4 段有 GT 录像不是 4 个已验证独立场地。
- 过去帧索引不证明标注产生过程在线因果。原标注含关键帧之间的插值。
- 仅知道 INI 写 25fps，不代表 M3W 的 seconds/metric 审计已经完成。
- 原始图像未下载，运动补偿、地面坐标、场地独立性与数据用途条款仍未解决。
- 测试录像的检测输出不能替代不存在的 GT。
- 当前准入守卫只保护该数据入口，不宣称历史所有脚本都已改造。

常规审计由项目执行并记录，不需要用户逐条确认。这里的不准入来自证据缺口，
不是等待用户替我检查。下一步应先落实标注来源、静态相机适用性及场地分组，
再登记任何预测实验，继续保留 DroneCrowd 的未读确认角色。
