"""POST /sdk/rollout-plan 测试：最少波次、字典序平局、已达目标跳过、blockers 与 422 校验。"""

from __future__ import annotations

import random

URL = "/sdk/rollout-plan"
TARGET = "digest-target"
OTHER = "digest-old"


def node(node_id, region="r1", failure_domain=None, weight=1, current=OTHER) -> dict:
    return {
        "node_id": node_id,
        "current_digest": current,
        "region": region,
        "failure_domain": failure_domain if failure_domain is not None else f"fd-{node_id}",
        "traffic_weight": weight,
    }


def limit(region="r1", weight=10**9) -> dict:
    return {"region": region, "max_unavailable_weight": weight}


def body(
    nodes: list[dict],
    *,
    max_parallel: int = 16,
    max_unavailable_weight: int = 10**9,
    region_limits: list[dict] | None = None,
    target_digest: str = TARGET,
) -> dict:
    regions = region_limits
    if regions is None:
        regions = [limit(r) for r in sorted({n["region"] for n in nodes})]
    return {
        "target_digest": target_digest,
        "nodes": nodes,
        "max_parallel": max_parallel,
        "max_unavailable_weight": max_unavailable_weight,
        "region_limits": regions,
    }


def plan(client, payload: dict) -> dict:
    response = client.post(URL, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def expect_422(client, payload: dict) -> None:
    response = client.post(URL, json=payload)
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# 基本规划与不变量
# ---------------------------------------------------------------------------


def test_single_wave(client):
    """全部约束宽松：所有待更新节点同一波完成。"""
    data = plan(client, body([node("a"), node("b"), node("c")], max_parallel=3))

    assert data["target_digest"] == TARGET
    assert data["status"] == "ready"
    assert data["blockers"] == []
    assert [w["node_ids"] for w in data["waves"]] == [["a", "b", "c"]]
    wave = data["waves"][0]
    assert wave["wave"] == 1
    assert wave["global_unavailable_weight"] == 3
    assert wave["region_unavailable_weights"] == [
        {"region": "r1", "unavailable_weight": 3}
    ]
    assert wave["completed_nodes"] == ["a", "b", "c"]


def test_all_nodes_at_target_is_empty_plan(client):
    """所有节点当前摘要即目标：零波完成，无 blocker。"""
    data = plan(
        client,
        body([node("a", current=TARGET), node("b", current=TARGET)], max_parallel=1),
    )

    assert data == {"target_digest": TARGET, "status": "ready", "waves": [], "blockers": []}


def test_nodes_already_at_target_are_skipped(client):
    """已达目标的节点不参与分波，也不影响 completed_nodes。"""
    data = plan(
        client,
        body(
            [
                node("a", current=TARGET),
                node("b"),
                node("c", current=TARGET),
                node("d"),
            ],
            max_parallel=1,
        ),
    )

    assert [w["node_ids"] for w in data["waves"]] == [["b"], ["d"]]
    assert data["waves"][-1]["completed_nodes"] == ["b", "d"]


def test_cumulative_completed_and_weights(client):
    """每波的累计完成节点为前缀并集；全局/分区不可用权重为本波小计。"""
    data = plan(
        client,
        body(
            [
                node("a", region="east", weight=3),
                node("b", region="west", weight=5),
                node("c", region="east", weight=7),
                node("d", region="west", weight=2),
            ],
            max_parallel=2,
            max_unavailable_weight=100,
            region_limits=[limit("east", 10), limit("west", 5)],
        )
    )

    assert len(data["waves"]) == 2
    first, second = data["waves"]
    assert first["completed_nodes"] == ["a", "b"]
    assert second["completed_nodes"] == ["a", "b", "c", "d"]
    # west 上限 5：b(5) 与 d(2) 必须分开；首波取字典序最小的 [a,b]。
    assert first["node_ids"] == ["a", "b"]
    assert first["global_unavailable_weight"] == 8
    assert first["region_unavailable_weights"] == [
        {"region": "east", "unavailable_weight": 3},
        {"region": "west", "unavailable_weight": 5},
    ]
    assert second["node_ids"] == ["c", "d"]
    assert second["global_unavailable_weight"] == 9
    assert second["region_unavailable_weights"] == [
        {"region": "east", "unavailable_weight": 7},
        {"region": "west", "unavailable_weight": 2},
    ]


# ---------------------------------------------------------------------------
# 最少波次与平局字典序
# ---------------------------------------------------------------------------


def test_minimum_waves_forced_by_concurrency(client):
    """max_parallel=2、4 个节点：最少 2 波，首波字典序最小取 [a,b]。"""
    data = plan(client, body([node("a"), node("b"), node("c"), node("d")], max_parallel=2))

    assert [w["node_ids"] for w in data["waves"]] == [["a", "b"], ["c", "d"]]


def test_tie_break_prefers_smaller_first_wave(client):
    """同为 2 波时，首波 ID 升序元组最小者胜——单元素 [a] 是 [a,b] 的前缀。"""
    data = plan(
        client,
        body([node("a"), node("b"), node("c")], max_parallel=3, max_unavailable_weight=2),
    )

    assert [w["node_ids"] for w in data["waves"]] == [["a"], ["b", "c"]]


def test_tie_break_between_equal_size_first_waves(client):
    """首波都为 2 个节点时，取第二元素更小者：[a,b] 先于 [a,c]。"""
    data = plan(
        client,
        body(
            [node("a", weight=2), node("b", weight=2), node("c", weight=2), node("d", weight=2)],
            max_parallel=2,
            max_unavailable_weight=4,
        ),
    )

    assert [w["node_ids"] for w in data["waves"]] == [["a", "b"], ["c", "d"]]


def test_global_weight_limit_forces_more_waves(client):
    """权重上限绑定：波内和不得超过全局上限，最少波次随容量收紧。"""
    nodes = [node("a", weight=4), node("b", weight=4), node("c", weight=4)]

    data = plan(client, body(nodes, max_parallel=3, max_unavailable_weight=8))
    # 2 波可行；字典序比较 ['a'] 是 ['a','b'] 的前缀，故首波取 ['a']。
    assert [w["node_ids"] for w in data["waves"]] == [["a"], ["b", "c"]]

    data = plan(client, body(nodes, max_parallel=3, max_unavailable_weight=4))
    assert [w["node_ids"] for w in data["waves"]] == [["a"], ["b"], ["c"]]


def test_region_limit_forces_split(client):
    """同分区两个权重 50 的节点，分区上限 50：即使全局允许也必须分波。"""
    data = plan(
        client,
        body(
            [node("a", weight=50), node("b", weight=50)],
            max_parallel=2,
            max_unavailable_weight=100,
            region_limits=[limit("r1", 50)],
        ),
    )

    assert [w["node_ids"] for w in data["waves"]] == [["a"], ["b"]]


def test_same_failure_domain_never_shares_wave(client):
    """同一 failure_domain 每波最多一个节点，即使其它约束宽松。"""
    data = plan(
        client,
        body(
            [
                node("a", failure_domain="rack-7"),
                node("b", failure_domain="rack-7"),
                node("c", failure_domain="rack-9"),
            ],
            max_parallel=3,
        ),
    )

    waves = [w["node_ids"] for w in data["waves"]]
    # ['a'] 是 ['a','c'] 的字典序前缀：首波尽量小，c 与 b 同波。
    assert waves == [["a"], ["b", "c"]]


def test_failure_domain_is_global_not_per_region(client):
    """failure_domain 跨 region 判重：同名域在不同分区仍须分波。"""
    data = plan(
        client,
        body(
            [
                node("a", region="east", failure_domain="same"),
                node("b", region="west", failure_domain="same"),
            ],
            max_parallel=2,
            region_limits=[limit("east"), limit("west")],
        ),
    )

    assert [w["node_ids"] for w in data["waves"]] == [["a"], ["b"]]


def test_string_ids_are_sorted_lexicographically_not_numerically(client):
    """平局按 ID 字符串字典序：'n10' 小于 'n2'。"""
    data = plan(
        client,
        body(
            [node("n10"), node("n2"), node("n3")],
            max_parallel=2,
        ),
    )

    # 字典序下 'n10' < 'n2' < 'n3'；['n10'] 又短于 ['n10','n2']。
    assert [w["node_ids"] for w in data["waves"]] == [["n10"], ["n2", "n3"]]


# ---------------------------------------------------------------------------
# blockers
# ---------------------------------------------------------------------------


def test_single_node_over_global_limit_is_blocker(client):
    """单节点独占一波仍超全局上限：blocked 且不出任何波。"""
    data = plan(
        client,
        body(
            [node("a", weight=100), node("b", weight=1)],
            max_unavailable_weight=50,
            region_limits=[limit("r1", 100)],
        ),
    )

    assert data["status"] == "blocked"
    assert data["waves"] == []
    assert data["blockers"] == [
        {
            "node_id": "a",
            "region": "r1",
            "failure_domain": "fd-a",
            "traffic_weight": 100,
            "reasons": ["exceeds_global_limit"],
        }
    ]


def test_single_node_over_region_limit_is_blocker(client):
    """分区上限小于节点权重：即使全局充足也 blocked。"""
    data = plan(
        client,
        body(
            [node("a", region="east", weight=10)],
            max_unavailable_weight=100,
            region_limits=[limit("east", 9)],
        ),
    )

    assert data["status"] == "blocked"
    assert data["blockers"][0]["reasons"] == ["exceeds_region_limit"]


def test_node_can_violate_both_limits(client):
    """同时超全局与分区上限：两个 reason 都给出，顺序为全局、分区。"""
    data = plan(
        client,
        body(
            [node("a", weight=10)],
            max_unavailable_weight=5,
            region_limits=[limit("r1", 5)],
        ),
    )

    assert data["blockers"][0]["reasons"] == [
        "exceeds_global_limit",
        "exceeds_region_limit",
    ]


def test_blockers_sorted_and_no_partial_plan(client):
    """多个 blocker 按节点 ID 升序；即使其余节点可规划也不返回任何波。"""
    data = plan(
        client,
        body(
            [
                node("a", weight=1),
                node("z", weight=99),
                node("m", weight=98),
            ],
            max_unavailable_weight=50,
            region_limits=[limit("r1", 100)],
        ),
    )

    assert data["status"] == "blocked"
    assert data["waves"] == []
    assert [b["node_id"] for b in data["blockers"]] == ["m", "z"]


def test_zero_region_limit_makes_pending_node_blocker(client):
    """分区上限 0 合法但意味着该分区任何待更新节点都无法起步。"""
    data = plan(
        client,
        body(
            [node("a", region="frozen", weight=1)],
            region_limits=[limit("frozen", 0)],
        ),
    )

    assert data["status"] == "blocked"
    assert data["blockers"][0]["reasons"] == ["exceeds_region_limit"]


def test_already_at_target_node_never_blocker(client):
    """已在目标摘要的节点跳过：即使权重超上限也不记为 blocker。"""
    data = plan(
        client,
        body(
            [node("a", weight=999, current=TARGET), node("b", weight=1)],
            max_unavailable_weight=50,
            region_limits=[limit("r1", 100)],
        ),
    )

    assert data["status"] == "ready"
    assert data["blockers"] == []
    assert [w["node_ids"] for w in data["waves"]] == [["b"]]


# ---------------------------------------------------------------------------
# 422 请求校验
# ---------------------------------------------------------------------------


def test_duplicate_node_id_422(client):
    expect_422(client, body([node("a"), node("a")]))


def test_unknown_region_limit_422(client):
    """节点所在分区未在 region_limits 声明 → 422。"""
    expect_422(
        client,
        body(
            [node("a", region="mystery")],
            region_limits=[limit("other")],
        ),
    )


def test_missing_region_limits_422(client):
    expect_422(client, body([node("a")], region_limits=[]))


def test_duplicate_region_limit_422(client):
    expect_422(
        client,
        body(
            [node("a", region="east")],
            region_limits=[limit("east", 10), limit("east", 20)],
        ),
    )


def test_non_positive_weight_422(client):
    expect_422(client, body([node("a", weight=0)]))
    expect_422(client, body([node("a", weight=-3)]))


def test_invalid_limits_422(client):
    expect_422(client, body([node("a")], max_parallel=0))
    expect_422(client, body([node("a")], max_parallel=-1))
    expect_422(client, body([node("a")], max_unavailable_weight=-1))
    expect_422(
        client,
        body([node("a")], region_limits=[{"region": "r1", "max_unavailable_weight": -1}]),
    )


def test_node_count_bounds_422_and_ok(client):
    expect_422(client, body([]))

    sixteen = [node(f"n{i}") for i in range(16)]
    data = plan(client, body(sixteen, max_parallel=16))
    assert len(data["waves"][0]["node_ids"]) == 16

    expect_422(client, body([node(f"n{i}") for i in range(17)]))


def test_empty_identifiers_and_extra_fields_422(client):
    expect_422(client, body([node("")]))
    expect_422(
        client,
        body(
            [node("a", region="", failure_domain="x")],
            region_limits=[limit(""), limit("x")],
        ),
    )
    expect_422(
        client,
        body(
            [node("a", failure_domain="")],
        ),
    )
    expect_422(client, {**body([node("a")]), "unexpected": 1})

    payload = body([node("a")])
    payload["nodes"][0]["rack"] = 7
    expect_422(client, payload)

    payload = body([node("a")])
    payload["region_limits"][0]["note"] = "x"
    expect_422(client, payload)


def test_empty_target_digest_422(client):
    expect_422(client, body([node("a")], target_digest=""))


def test_extra_unused_region_limit_is_allowed(client):
    """声明了但没有节点使用的分区限额无害，不拒绝。"""
    data = plan(
        client,
        body([node("a", region="east")], region_limits=[limit("east"), limit("west")]),
    )
    assert data["status"] == "ready"


# ---------------------------------------------------------------------------
# 穷举交叉验证：最优波数与字典序平局
# ---------------------------------------------------------------------------


def _brute_force(nodes: list[dict], max_parallel: int, global_limit: int, region_limits: dict):
    """参考实现：记忆化递归穷举所有分波，返回最少波数下字典序最小的 ID 序列。"""
    ordered = sorted(nodes, key=lambda n: n["node_id"])
    n = len(ordered)
    weights = [x["traffic_weight"] for x in ordered]
    regions = [x["region"] for x in ordered]
    domains = [x["failure_domain"] for x in ordered]

    def feasible(mask: int) -> bool:
        bits = [i for i in range(n) if mask >> i & 1]
        if len(bits) > max_parallel:
            return False
        if sum(weights[i] for i in bits) > global_limit:
            return False
        if len({domains[i] for i in bits}) != len(bits):
            return False
        by_region: dict[str, int] = {}
        for i in bits:
            by_region[regions[i]] = by_region.get(regions[i], 0) + weights[i]
        return all(weight <= region_limits[r] for r, weight in by_region.items())

    memo: dict[int, tuple[int, tuple]] = {}

    def solve(mask: int) -> tuple[int, tuple]:
        if mask == 0:
            return 0, ()
        if mask in memo:
            return memo[mask]
        lowest = mask & -mask
        rest = mask ^ lowest
        best = None
        sub = rest
        while True:
            wave = sub | lowest
            if feasible(wave):
                remaining = solve(rest ^ sub)
                if remaining is None:
                    if sub == 0:
                        break
                    sub = (sub - 1) & rest
                    continue
                count, tail = remaining
                ids = tuple(ordered[i]["node_id"] for i in range(n) if wave >> i & 1)
                candidate = (count + 1, (ids,) + tail)
                if best is None or candidate < best:
                    best = candidate
            if sub == 0:
                break
            sub = (sub - 1) & rest
        memo[mask] = best
        return best

    return solve((1 << n) - 1)


def test_matches_brute_force_on_random_instances(client):
    """随机小实例上服务结果须与穷举参考（最少波数 + 字典序最小）完全一致。"""
    rng = random.Random(20260929)
    checked = 0
    for _ in range(400):
        n = rng.randint(1, 6)
        ids = [f"n{i}" for i in range(n)]
        regions = [f"r{rng.randrange(3)}" for _ in range(n)]
        domains = [f"fd{rng.randrange(4)}" for _ in range(n)]
        weights = [rng.randint(1, 4) for _ in range(n)]
        nodes = [
            node(ids[i], region=regions[i], failure_domain=domains[i], weight=weights[i])
            for i in range(n)
        ]
        max_parallel = rng.randint(1, n)
        global_limit = rng.randint(1, 10)
        region_limits = {f"r{r}": rng.randint(1, 8) for r in range(3)}

        result = _brute_force(
            nodes, max_parallel, global_limit, region_limits
        )

        # 参考解不可行（存在单节点 blocker）时跳过结构比对，仅核对 blocked。
        single_feasible = all(
            w <= global_limit and w <= region_limits[r]
            for w, r in zip(weights, regions)
        )
        if not single_feasible:
            assert result is None
            data = plan(
                client,
                body(
                    nodes,
                    max_parallel=max_parallel,
                    max_unavailable_weight=global_limit,
                    region_limits=[limit(r, lim) for r, lim in sorted(region_limits.items())],
                ),
            )
            assert data["status"] == "blocked"
            assert data["waves"] == []
            blocker_ids = {b["node_id"] for b in data["blockers"]}
            assert blocker_ids == {
                ids[i]
                for i in range(n)
                if weights[i] > global_limit or weights[i] > region_limits[regions[i]]
            }
            checked += 1
            continue

        count, expected_seq = result
        assert expected_seq is not None
        data = plan(
            client,
            body(
                nodes,
                max_parallel=max_parallel,
                max_unavailable_weight=global_limit,
                region_limits=[limit(r, lim) for r, lim in sorted(region_limits.items())],
            ),
        )
        assert data["status"] == "ready"
        actual_seq = tuple(tuple(w["node_ids"]) for w in data["waves"])
        assert len(actual_seq) == count
        assert actual_seq == expected_seq
        checked += 1

    assert checked == 400


def test_wave_invariants_on_random_instances(client):
    """随机实例上逐波核对并发、全局/分区权重与 failure_domain 约束。"""
    rng = random.Random(4242)
    for _ in range(100):
        n = rng.randint(1, 10)
        nodes = [
            node(
                f"n{i}",
                region=f"r{rng.randrange(2)}",
                failure_domain=f"fd{rng.randrange(5)}",
                weight=rng.randint(1, 6),
            )
            for i in range(n)
        ]
        max_parallel = rng.randint(1, 4)
        global_limit = rng.randint(6, 20)
        limits = {f"r{r}": rng.randint(6, 15) for r in range(2)}

        data = plan(
            client,
            body(
                nodes,
                max_parallel=max_parallel,
                max_unavailable_weight=global_limit,
                region_limits=[limit(r, lim) for r, lim in sorted(limits.items())],
            ),
        )
        if data["status"] == "blocked":
            assert data["waves"] == []
            continue

        by_id = {x["node_id"]: x for x in nodes}
        seen: set[str] = set()
        for wave in data["waves"]:
            members = wave["node_ids"]
            assert len(members) <= max_parallel
            assert len(members) == len(set(members))
            domains = [by_id[m]["failure_domain"] for m in members]
            assert len(domains) == len(set(domains))
            total = sum(by_id[m]["traffic_weight"] for m in members)
            assert total <= global_limit
            assert wave["global_unavailable_weight"] == total
            by_region: dict[str, int] = {}
            for m in members:
                x = by_id[m]
                by_region[x["region"]] = by_region.get(x["region"], 0) + x["traffic_weight"]
            assert [r["region"] for r in wave["region_unavailable_weights"]] == sorted(by_region)
            for entry in wave["region_unavailable_weights"]:
                assert entry["unavailable_weight"] == by_region[entry["region"]]
                assert entry["unavailable_weight"] <= limits[entry["region"]]
            seen.update(members)
            assert wave["completed_nodes"] == sorted(seen)
        assert seen == {x["node_id"] for x in nodes}
