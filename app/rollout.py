"""SDK 配置制品分批铺开：在不可用权重与并发约束下精确求最少更新波次。

语义概述：

- 只规划 ``current_digest`` 与 ``target_digest`` 不同的节点，已在目标摘要的
  节点跳过（不计入任何波）；
- 同一波节点同时不可用，因此每个波（节点子集）必须同时满足：

  * 波内节点数 ≤ ``max_parallel``；
  * 波内流量权重和 ≤ 全局 ``max_unavailable_weight``；
  * 各 region 的波内权重和 ≤ 该 region 的分区上限；
  * 同一 ``failure_domain`` 每波最多出现一个节点（failure_domain 全局判重，
    不按 region 切分）。

- 任一节点单独更新（独占一波）即超全局或其分区上限时无法进入任何波，
  记为 blocker：存在 blocker 时不给出任何部分计划；
- 无 blocker 时在所有可行分波方案中求**最少波次**；多解时把每波节点 ID
  升序排列后，取波次序列**字典序最小**者。

求解（节点数 ≤ 16，在 2^N 子集空间精确求解）：

1. O(2^N) 低位递推预计算每个子集的权重和、failure_domain 占用位与各分区
   权重和，并判定其能否作为同一波；同时给出波数下界（并发/全局/分区/
   同域冲突四个维度上取整容量的最大值）；
2. 子集 DP：``waves[mask]`` 为覆盖 mask 的最少波数。去重约束“首波包含
   mask 最低位节点”，只枚举以该位为最低位的子掩码；整集本身可行即 1 波，
   下界等于节点数即只能全部独占，其余情形按子掩码从大到小枚举并在拿到
   下界值时提前结束；
3. 贪心重构字典序最小的波次序列：每一步在“含剩余集合最低位节点”的可行
   波中，取成员升序元组最小（Python tuple 序，含短前缀规则，注意它不等价
   于波掩码数值序）且剩余节点恰能用剩余波数覆盖的那一波。
"""

from __future__ import annotations

from array import array
from math import ceil

from app.models import (
    RegionUnavailableWeight,
    RolloutBlocker,
    RolloutBlockerReason,
    RolloutPlanRequest,
    RolloutPlanResponse,
    RolloutStatus,
    RolloutWave,
)


def _members(mask: int) -> tuple[int, ...]:
    """返回 mask 中各节点位（按位序升序），与节点 ID 升序一致。"""
    bits: list[int] = []
    while mask:
        lowest = mask & -mask
        bits.append(lowest.bit_length() - 1)
        mask ^= lowest
    return tuple(bits)


def build_rollout_plan(request: RolloutPlanRequest) -> RolloutPlanResponse:
    """生成最少波次铺开计划；存在单独更新即超限的节点时只返回 blockers。"""
    pending = [
        node
        for node in sorted(request.nodes, key=lambda n: n.node_id)
        if node.current_digest != request.target_digest
    ]

    region_limit = {
        entry.region: entry.max_unavailable_weight for entry in request.region_limits
    }

    # 先找 blocker：节点独占一波仍超全局或分区上限（failure_domain 与并发
    # 对单节点波天然满足，max_parallel ≥ 1 已由模型保证）。
    blockers: list[RolloutBlocker] = []
    for node in pending:
        reasons: list[RolloutBlockerReason] = []
        if node.traffic_weight > request.max_unavailable_weight:
            reasons.append(RolloutBlockerReason.EXCEEDS_GLOBAL_LIMIT)
        if node.traffic_weight > region_limit[node.region]:
            reasons.append(RolloutBlockerReason.EXCEEDS_REGION_LIMIT)
        if reasons:
            blockers.append(
                RolloutBlocker(
                    node_id=node.node_id,
                    region=node.region,
                    failure_domain=node.failure_domain,
                    traffic_weight=node.traffic_weight,
                    reasons=reasons,
                )
            )
    if blockers:
        # 有 blocker 时不得给出部分计划：waves 恒为空。
        return RolloutPlanResponse(
            target_digest=request.target_digest,
            status=RolloutStatus.BLOCKED,
            blockers=blockers,
        )

    if not pending:
        # 全部节点已在目标摘要：零波完成。
        return RolloutPlanResponse(
            target_digest=request.target_digest,
            status=RolloutStatus.READY,
        )

    size = len(pending)
    all_mask = (1 << size) - 1
    table_size = all_mask + 1
    weights = [node.traffic_weight for node in pending]
    regions = [node.region for node in pending]
    domain_index = {
        domain: i
        for i, domain in enumerate(sorted({n.failure_domain for n in pending}))
    }
    node_domain_bit = [1 << domain_index[node.failure_domain] for node in pending]
    # 每个 failure_domain 包含哪些节点，用于求“单域节点数”这一硬下界。
    domain_members = [0] * len(domain_index)
    for i, domain_bit in enumerate(node_domain_bit):
        domain_members[domain_bit.bit_length() - 1] |= 1 << i

    # 低位递推：每个子集的权重和、failure_domain 位掩码与各分区权重和。
    total = array("q", [0]) * table_size
    fd_mask = array("q", [0]) * table_size
    region_sums: dict[str, array] = {
        region: array("q", [0]) * table_size for region in region_limit
    }
    for mask in range(1, table_size):
        lowest = mask & -mask
        i = lowest.bit_length() - 1
        rest = mask ^ lowest
        total[mask] = total[rest] + weights[i]
        fd_mask[mask] = fd_mask[rest] | node_domain_bit[i]
        for region, sums in region_sums.items():
            sums[mask] = sums[rest] + (weights[i] if regions[i] == region else 0)

    feasible = bytearray(table_size)
    lower_bound = array("b", bytes(size + 1)) * table_size
    for mask in range(1, table_size):
        count = mask.bit_count()
        ok = (
            count <= request.max_parallel
            and total[mask] <= request.max_unavailable_weight
            # 同一 failure_domain 在波内最多一个节点：域出现数须等于节点数。
            and fd_mask[mask].bit_count() == count
        )
        bound = max(
            ceil(count / request.max_parallel),
            ceil(total[mask] / max(request.max_unavailable_weight, 1)),
            1,
        )
        for region, limit in region_limit.items():
            sums = region_sums[region][mask]
            if sums > limit:
                ok = False
            if sums:
                bound = max(bound, ceil(sums / max(limit, 1)))
        if ok:
            feasible[mask] = 1
        # 每个 failure_domain 的节点两两不能同波：该域在 mask 中的节点数即硬下界。
        for member_mask in domain_members:
            domain_count = (mask & member_mask).bit_count()
            if domain_count > bound:
                bound = domain_count
        lower_bound[mask] = bound

    # 子集 DP：waves[mask] = 把 mask 中节点分波更新所需的最少波数。
    waves = array("b", bytes(size + 1)) * table_size
    waves[0] = 0
    for mask in range(1, table_size):
        if feasible[mask]:
            # 整集一波可行，即最小波数 1。
            waves[mask] = 1
            continue
        bound = lower_bound[mask]
        if bound == mask.bit_count():
            # 下界等于节点数：只能每个节点独占一波。
            waves[mask] = bound
            continue
        lowest = mask & -mask
        rest = mask ^ lowest
        best = size + 1
        # 首波必含 lowest；sub 从大到小枚举即首波从大到小，容易率先达到
        # 下界从而提前结束。
        sub = rest
        while True:
            wave = sub | lowest
            if feasible[wave]:
                candidate = waves[mask ^ wave] + 1
                if candidate < best:
                    best = candidate
                    if best == bound:
                        break
            if sub == 0:
                break
            sub = (sub - 1) & rest
        waves[mask] = best

    needed = waves[all_mask]
    # 所有单节点波均可行（无 blocker），全量必然可分。
    assert 1 <= needed <= size

    # 贪心重构字典序最小的波次序列：每一步在“含剩余集合最低位节点”的
    # 可行波中，挑选使完整剩余序列字典序最小者。由于最优子结构，本步只需
    # 取**成员升序元组最小**且剩余节点恰能用剩余波数覆盖的波。
    # 成员元组按 Python tuple 比较：('n0','n1','n2') < ('n0','n2')
    # （第二位 'n1'<'n2'），('n0',) < ('n0','n1')（短前缀更小）。
    # 注意它不等价于波掩码数值序（0b111=7 成员元组 (0,1,2) 小于
    # 0b101=5 的 (0,2)），故须显式比较成员元组。
    result_waves: list[RolloutWave] = []
    remaining = all_mask
    completed: set[str] = set()
    wave_index = 0
    while remaining:
        wave_index += 1
        lowest = remaining & -remaining
        rest = remaining ^ lowest
        chosen_bits: tuple[int, ...] | None = None
        sub = rest
        while True:
            wave = sub | lowest
            if feasible[wave] and waves[remaining ^ wave] == needed - wave_index:
                candidate_bits = _members(wave)
                if chosen_bits is None or candidate_bits < chosen_bits:
                    chosen_bits = candidate_bits
            if sub == 0:
                break
            sub = (sub - 1) & rest
        assert chosen_bits is not None
        chosen_mask = 0
        for i in chosen_bits:
            chosen_mask |= 1 << i

        by_region: dict[str, int] = {}
        node_ids: list[str] = []
        for i in chosen_bits:
            node = pending[i]
            node_ids.append(node.node_id)
            by_region[node.region] = by_region.get(node.region, 0) + node.traffic_weight
        completed.update(node_ids)

        result_waves.append(
            RolloutWave(
                wave=wave_index,
                node_ids=node_ids,
                global_unavailable_weight=total[chosen_mask],
                region_unavailable_weights=[
                    RegionUnavailableWeight(
                        region=region, unavailable_weight=weight
                    )
                    for region, weight in sorted(by_region.items())
                ],
                completed_nodes=sorted(completed),
            )
        )
        remaining ^= chosen_mask

    return RolloutPlanResponse(
        target_digest=request.target_digest,
        status=RolloutStatus.READY,
        waves=result_waves,
    )
