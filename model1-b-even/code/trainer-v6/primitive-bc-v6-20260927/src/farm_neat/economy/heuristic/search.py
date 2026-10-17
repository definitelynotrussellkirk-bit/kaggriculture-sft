"""Small exact search routines used by the heuristic's receding-horizon planner.

These routines deliberately solve only models we can state faithfully inside the submission.
They do not simulate hidden opponent actions or stochastic end-of-day events.
"""

from collections import deque
from functools import lru_cache

from .constants import MOVES


def _apply_job_inventory(inventory, job):
    """Return inventory after ``job``, or ``None`` when its item precondition fails.

    The horizon planner intentionally models only effects that can change which later job orders
    are legal.  This is enough to distinguish, for example, PICKUP->FEED from the impossible
    FEED->PICKUP ordering without copying the whole game engine into the submission.
    """

    result = dict(inventory)
    for item, quantity in job.get("consume", {}).items():
        if result.get(item, 0) < quantity:
            return None
        result[item] -= quantity
        if result[item] == 0:
            del result[item]
    for item, quantity in job.get("gain", {}).items():
        result[item] = result.get(item, 0) + quantity
    return result


def enumerate_job_schedules(
    source,
    inventory,
    jobs,
    distance,
    horizon,
    *,
    action_cost=0.0,
    candidate_limit=12,
    schedule_limit=64,
    node_limit=8_000,
):
    """Enumerate high-value legal job orders for one unit over a bounded horizon.

    ``distance(a, b)`` supplies exact shortest-path lengths.  A job consumes one turn at its
    position and may provide ``consume``, ``gain``, ``deadline`` and ``invalidates_position``
    fields.  The candidate shortlist is deterministic and deliberately bounded; within that
    shortlist every feasible permutation is visited until ``node_limit``.  That makes order
    sensitivity explicit while keeping a 1-second competition action deadline safe.

    Returned schedules contain a global-job bit mask, ordered job indices, completion time and
    net utility (job dollars less the configured shadow cost of movement/actions).  The empty
    schedule is always present.
    """

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if candidate_limit < 1 or schedule_limit < 1 or node_limit < 1:
        raise ValueError("search limits must be positive")

    empty = {"jobs": (), "mask": 0, "score": 0.0, "time": 0}
    if horizon == 0 or not jobs:
        return [empty], {"nodes": 0, "truncated": False, "candidates": 0}

    # Do not let a large farm make enumeration factorial in every currently visible job.  Rank
    # the unit-specific shortlist by attainable value density, but retain infeasible item jobs:
    # an earlier PICKUP/HARVEST/COLLECT may make those feasible later in the same sequence.
    ranked = []
    for index, job in enumerate(jobs):
        travel = distance(tuple(source), tuple(job["pos"]))
        if travel is None:
            continue
        completion = travel + 1
        deadline = min(horizon, int(job.get("deadline", horizon)))
        if completion > deadline:
            continue
        density = float(job["value"]) / completion
        ranked.append((-density, completion, index))
    shortlist = [index for _density, _time, index in sorted(ranked)[:candidate_limit]]
    if not shortlist:
        return [empty], {"nodes": 0, "truncated": False, "candidates": 0}

    schedules = [empty]
    # Every directly executable job remains an assignment edge. The bounded permutation
    # shortlist controls only continuation branching; using it as the first-action shortlist
    # made many workers compete for the same eight jobs and idle despite other useful work.
    for _density, completion, index in sorted(ranked):
        job = jobs[index]
        if _apply_job_inventory(inventory, job) is None:
            continue
        schedules.append({
            "jobs": (index,),
            "mask": 1 << index,
            "score": (
                float(job["value"]) / completion
                - action_cost * completion / max(1, completion)
            ),
            "time": completion,
        })
    frontier = [(tuple(source), 0, dict(inventory), (), 0, 0.0, frozenset())]
    nodes = 0
    truncated = False

    while frontier:
        position, elapsed, held, sequence, used_mask, score, invalidated = frontier.pop()
        for index in shortlist:
            bit = 1 << index
            if used_mask & bit:
                continue
            job = jobs[index]
            target = tuple(job["pos"])
            if target in invalidated:
                continue
            travel = distance(position, target)
            if travel is None:
                continue
            completion = elapsed + travel + 1
            deadline = min(horizon, int(job.get("deadline", horizon)))
            if completion > deadline:
                continue
            next_held = _apply_job_inventory(held, job)
            if next_held is None:
                continue

            nodes += 1
            if nodes > node_limit:
                truncated = True
                frontier.clear()
                break
            next_sequence = sequence + (index,)
            next_mask = used_mask | bit
            # Receding-horizon utility: a dollar completed sooner can be reinvested and frees the
            # worker for another observed job. For the first job this is exactly the production
            # policy's proven value-density objective; later rewards are discounted by their
            # absolute completion time, making this its natural multi-turn extension.
            next_score = (
                score
                + float(job["value"]) / completion
                - action_cost * (travel + 1) / max(1, completion)
            )
            next_invalidated = invalidated
            if job.get("invalidates_position"):
                next_invalidated = invalidated | {target}
            schedule = {
                "jobs": next_sequence,
                "mask": next_mask,
                "score": next_score,
                "time": completion,
            }
            schedules.append(schedule)
            frontier.append((
                target,
                completion,
                next_held,
                next_sequence,
                next_mask,
                next_score,
                next_invalidated,
            ))

    # A sequence with the same job set but a slower completion can never help the joint solver.
    # Keep the best representative, then retain a broad mix of absolute value and value density.
    best_by_mask = {}
    for schedule in schedules:
        prior = best_by_mask.get(schedule["mask"])
        key = (schedule["score"], -schedule["time"], tuple(-j for j in schedule["jobs"]))
        if prior is None:
            best_by_mask[schedule["mask"]] = schedule
            continue
        prior_key = (prior["score"], -prior["time"], tuple(-j for j in prior["jobs"]))
        if key > prior_key:
            best_by_mask[schedule["mask"]] = schedule
    nonempty = [s for mask, s in best_by_mask.items() if mask]
    by_value = sorted(nonempty, key=lambda s: (-s["score"], s["time"], s["jobs"]))
    by_density = sorted(
        nonempty,
        key=lambda s: (-(s["score"] / max(1, s["time"])), -s["score"], s["jobs"]),
    )
    # Preserve at least one rollout for every possible first action; schedule_limit applies to
    # additional alternate multi-job sets, not to the executable action space.
    best_by_first = {}
    for schedule in nonempty:
        first = schedule["jobs"][0]
        prior = best_by_first.get(first)
        if prior is None or (schedule["score"], -schedule["time"], schedule["jobs"]) > (
            prior["score"], -prior["time"], prior["jobs"]
        ):
            best_by_first[first] = schedule
    kept = sorted(
        best_by_first.values(),
        key=lambda s: (-s["score"], s["time"], s["jobs"]),
    )
    seen = set()
    seen.update(s["mask"] for s in kept)
    for schedule in by_value[:schedule_limit] + by_density[:schedule_limit]:
        if schedule["mask"] in seen:
            continue
        kept.append(schedule)
        seen.add(schedule["mask"])
        if len(kept) >= len(best_by_first) + schedule_limit:
            break
    kept.sort(key=lambda s: (-s["score"], s["time"], s["jobs"]))
    return [empty, *kept], {
        "nodes": nodes,
        "truncated": truncated,
        "candidates": len(shortlist),
    }


def select_joint_schedules(schedule_sets, *, node_limit=60_000):
    """Branch-and-bound over one schedule per unit with globally exclusive jobs.

    Per-unit schedules have already enumerated alternate 5--10 turn orders.  This second search
    resolves their shared-job conflicts.  It is exact whenever the deterministic node budget is
    not exhausted and returns the best incumbent otherwise, so runtime has a hard bound.
    """

    if node_limit < 1:
        raise ValueError("node_limit must be positive")
    if not schedule_sets:
        return {}, {"nodes": 0, "truncated": False}

    normalized = []
    for unit, schedules in enumerate(schedule_sets):
        choices = sorted(
            schedules,
            key=lambda s: (-float(s["score"]), int(s["time"]), tuple(s["jobs"])),
        )
        normalized.append((unit, choices))
    # Put the units with the most valuable/constrained choices first so good incumbents and
    # conflicts appear early. This makes the upper bound effective without changing the answer.
    normalized.sort(key=lambda row: (-row[1][0]["score"], len(row[1]), row[0]))
    upper = [0.0] * (len(normalized) + 1)
    for depth in range(len(normalized) - 1, -1, -1):
        upper[depth] = upper[depth + 1] + max(0.0, normalized[depth][1][0]["score"])

    best_score = -1.0
    best_choice = {}
    nodes = 0
    truncated = False

    def visit(depth, used, score, chosen):
        nonlocal best_score, best_choice, nodes, truncated
        if score + upper[depth] <= best_score + 1e-12:
            return
        if depth == len(normalized):
            if score > best_score + 1e-12:
                best_score = score
                best_choice = dict(chosen)
            return
        unit, choices = normalized[depth]
        for schedule in choices:
            if schedule["mask"] & used:
                continue
            nodes += 1
            if nodes > node_limit:
                truncated = True
                return
            chosen[unit] = schedule
            visit(depth + 1, used | schedule["mask"], score + schedule["score"], chosen)
            if truncated:
                return
        chosen.pop(unit, None)

    visit(0, 0, 0.0, {})
    return best_choice, {"nodes": nodes, "truncated": truncated, "score": best_score}


@lru_cache(maxsize=512)
def _rectangular_shortest_routes(board, source):
    """Cached BFS for the engine's obstacle-free square movement topology.

    Keep the same ``MOVES`` traversal as the generic implementation below: equal-distance route
    ties therefore select the exact same first operation as before the cache existed.  Callers
    receive a copy so the public result remains an independently mutable dictionary.
    """

    routes = {tuple(source): (0, None)}
    queue = deque([tuple(source)])
    while queue:
        position = queue.popleft()
        distance, first = routes[position]
        for operation, (dx, dy) in MOVES.items():
            nxt = (position[0] + dx, position[1] + dy)
            if (
                nxt in routes
                or not (0 <= nxt[0] < board and 0 <= nxt[1] < board)
            ):
                continue
            routes[nxt] = (distance + 1, first or operation)
            queue.append(nxt)
    return routes


def shortest_routes(view, source):
    """Return ``position -> (distance, first_move)`` for the current walkable board.

    The board is tiny, so a complete breadth-first search per available unit is cheaper and more
    reliable than Manhattan routing.  ``first_move`` is ``None`` at the source itself.
    """

    source = tuple(source)
    if getattr(view, "all_board_cells_walkable", False):
        board = int(view.board)
        if board <= 0:
            raise ValueError("walkable board size must be positive")
        return dict(_rectangular_shortest_routes(board, source))

    routes = {source: (0, None)}
    queue = deque([source])
    while queue:
        position = queue.popleft()
        distance, first = routes[position]
        for operation, (dx, dy) in MOVES.items():
            nxt = (position[0] + dx, position[1] + dy)
            if nxt in routes or not view.walkable(*nxt):
                continue
            routes[nxt] = (distance + 1, first or operation)
            queue.append(nxt)
    return routes


def maximum_weight_assignment(unit_count, job_count, edges):
    """Solve the complete one-unit/one-job assignment by augmenting-path search.

    ``edges[(unit, job)] = (utility, distance)``. Missing or non-positive edges are infeasible;
    units and jobs may remain idle. A rectangular Hungarian search over real jobs plus one dummy
    idle column per unit gives the global optimum in ``O(n_units**2 * (n_jobs+n_units))`` with a
    deterministic worst-case bound.

    Returns ``{unit_index: job_index}``.
    """

    if unit_count < 0 or job_count < 0:
        raise ValueError("unit_count and job_count must be non-negative")
    if unit_count == 0:
        return {}

    column_count = job_count + unit_count
    forbidden = 1e15

    def cost(unit, column):
        if column >= job_count:  # distinct zero-value idle choices
            return 0.0
        edge = edges.get((unit, column))
        if edge is None or edge[0] <= 0:
            return forbidden
        utility, distance = edge
        # Distance and stable indices are tie-breakers only; utility remains the objective.
        return -float(utility) + int(distance) * 1e-10 + column * 1e-13

    # Standard rectangular Hungarian augmenting-path search, 1-indexed internally. There are at
    # least as many columns as rows because every unit receives its own idle column.
    row_potential = [0.0] * (unit_count + 1)
    column_potential = [0.0] * (column_count + 1)
    column_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, unit_count + 1):
        column_row[0] = row
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            active_row = column_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost(active_row - 1, candidate_column - 1)
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    previous_column[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[column_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            column_row[column] = column_row[prior]
            column = prior
            if column == 0:
                break

    assigned = {}
    for column in range(1, column_count + 1):
        row = column_row[column]
        job = column - 1
        if row and job < job_count:
            edge = edges.get((row - 1, job))
            if edge is not None and edge[0] > 0:
                assigned[row - 1] = job
    return assigned
