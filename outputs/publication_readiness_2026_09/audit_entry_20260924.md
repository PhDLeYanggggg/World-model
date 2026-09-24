# M3W 审计入口：2026-09-24

## 一句话结论

本轮真实神经训练、交叉拟合和复核已完成。**平均预测有正信号，但安全门槛失败，联合决策贡献没有成立，不升级部署，也不能称 CVPR/A会投稿候选已达成。**

当前实验是 European Squares 的 source-only 开发实验，不是之前 Stage37 的 t+50 协议。这里观察8步、预测12步，原始帧间隔12，图像像素坐标。不能把两套数字直接比较。

## 建议先查这六项

| 审计问题 | 结果 | 证据 |
|---|---|---|
| 真训练还是只读旧报告？ | 18个Transformer各4,000更新；9个神经风险头各2,000更新；另9个ridge闭式拟合 | [训练日志表](european_source_forecast_v1/training_losses.md)、[风险头训练表](european_source_intervention_v1/training_losses.md) |
| 预测是否真的改善？ | 对训练地点选出的基线，平均ADE改善4.1135%，条件性地点bootstrap CI[1.3672%,6.9485%] | [完整预测表](european_source_forecast_v1/results.md) |
| 是否击败所有强基线？ | 没有证明。同一CV参照下，神经网络+2.1576%，固定damping0.97为+3.9755% | 同上，包含所有固定基线 |
| easy和零误差样本是否安全？ | 失败。easy退化13.65%–14.39%，3个seed都伤害全部4个零CV误差样本 | [失败结论](european_source_forecast_v1/conclusions.md) |
| 联合决策是不是有效创新？ | 尚未成立。神经风险头在相同切换人数下，joint对independent的改善3个seed都是0 | [配对对照及CI](european_source_intervention_v1/paired_joint_contrasts.md) |
| 能否复现？ | 全部指标缓存重算一致；18个预测检查点各抽128行、18个风险头各抽4,096行重新推理完全一致；127项相关测试通过 | [复现命令与边界](european_source_execution_20260924.md) |

## 最重要的失败原因

1. **fallback选错了安全参照。** 它按训练地点平均误差选出，却相对CV造成15.48%的easy退化。预测“不比这个fallback差”不代表满足原定的“easy相对CV退化≤2%”。这不是多加一个阈值就自动解决的问题。
2. **平均误差目标不等于安全约束。** 轨迹模型优化平均ADE；风险头估计正增益和正伤害。两者都没有通过独立校准证明easy/零误差约束。4倍惩罚伤害低估也不是统计上界。
3. **联合项没有稳定额外价值。** 6,116目标的联合决策试验中，独立和联合规则基本相同；ridge有的配对结果反而更差。降低预测轨迹重叠代理不能代替真实预测贡献或物理碰撞安全。
4. **泛化还有明显差异。** 12地点中3个平均变差；最大地点的改善是负的。风险分数在部分地点与真实增益相关性很弱。教师训练规模变化可能参与其中，但本轮没有隔离证明其因果作用。

## 结果来源，不混用

- `fresh_run`：本轮18个轨迹模型训练、新预测、18个风险头拟合、固定联合策略求解和第一次统一评价。
- `cached_verified`：已做过完整原始数据重放的source cache，本轮重新核验hash/schema/roles；已保存预测与决策的全量指标重算。不是第二次训练。
- `fresh_run`检查点复核：保存权重重新推理指定行，输出逐项一致。仅为抽查，不虚称全量重新推理或独立团队复现。
- `not_run`：保留的6个模型选择地点、12个风险校准地点、6个确认地点的模型评价；DroneCrowd确认评价；新模型部署；正式论文提交；Stage5C和SMC。

## 数据与泄露边界

本轮只开放12个训练地点、163个recordings，共318,969个完整过去窗口目标。311,922行至少有一个未来标签，7,047行全部未来标签缺失，仍保留在推理母体中。图像坐标来自发布的检测轨迹，不是人工gold。

外层地点同时被排除出预测器、标签生成教师、归一化和风险头训练。未来坐标/有效掩码不进入模型输入；没有用test endpoint建goal。训练之外的213个recordings没有开放给模型。分组去重审计的通过不等于已证明世界范围数据独立，也不证明检测器在线因果性。

bootstrap用地点而非相互重叠的窗口，3,000次；模型共享训练来源，所以这是条件性开发区间，不是独立确认。三个seed不能冒充三个独立数据集。

## 历史数字需要降级理解

早期Stage37的“16/16”和“deployable”标签不能继续当作干净泛化证据：后续[录制来源审计](stage35_recording_lineage_audit.md)发现47,223个相同窗口跨val/test，并发现Stage37用test指标排列selector版本。旧结果保留为探索记录，不据此宣称当前已获安全部署或外部确认。

## 下一步最短路径

先版本化修复CV-relative安全参照和风险目标，保留固定damping/历史OLS强对照，再检验是否获得**平均改善且easy不受损**的源内候选。联合几何项在有稳定配对增益前只能作为负对照，不能充当论文主贡献。方法冻结后才考虑按角色开放模型选择和独立风险校准；确认数据最后使用。

本轮没有根据读出的结果重选阈值、换主指标、删掉坏地点或挑最好seed。没有改动已完成实验的身份绑定文件。README与research_state会反映失败，不使用“门槛全过”的总分掩盖关键失败。

## 查看顺序

1. [预测结论与全部负结果](european_source_forecast_v1/conclusions.md)
2. [风险头和联合对照结论](european_source_intervention_v1/conclusions.md)
3. [执行/恢复/复核命令](european_source_execution_20260924.md)
4. [方法公式与不能声称的内容](european_source_intervention_v1/method_draft.md)
5. [预测原始轻量指标](european_source_forecast_v1/analysis.json)及[风险头指标](european_source_intervention_v1/analysis.json)
