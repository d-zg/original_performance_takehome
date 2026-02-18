"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""
from collections import Counter, defaultdict, deque
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class VReg:
    """A virtual register representing a value, not a physical location."""
    _counter = 0

    def __init__(self, name_hint="", size=1, pinned_addr=None):
        self.id = VReg._counter
        VReg._counter += 1
        self.name_hint = name_hint
        self.size = size  # 1 for scalar, VLEN for vector
        self.pinned_addr = pinned_addr  # If set, must use this physical address
        self.parent = None  # If set, this is a sub-element of a vector vreg
        self.offset = 0     # Offset within parent

    def __repr__(self):
        if self.name_hint:
            return f"v{self.id}_{self.name_hint}"
        return f"v{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, VReg):
            return self.id == other.id
        return False


def get_defs_uses(engine, args):
    """Extract VReg defs (writes) and uses (reads) from a slot.

    Returns (defs: set[VReg], uses: set[VReg]).
    Tracks ALL VRegs including pinned ones for dependency analysis.
    """
    if engine == "debug":
        return set(), set()

    if engine == "store":
        # Stores write to main memory, not scratch. All VReg args are uses.
        uses = {a for a in args[1:] if isinstance(a, VReg)}
        return set(), uses

    # For alu, valu, load: args[1] is dest (def), args[2:] are sources (uses)
    defs = set()
    uses = set()
    if len(args) > 1 and isinstance(args[1], VReg):
        defs.add(args[1])
    for a in args[2:]:
        if isinstance(a, VReg):
            uses.add(a)
    return defs, uses


class ScratchAllocator:
    """Manages scratch space allocation for virtual registers.

    Vectors (size VLEN) scan left-to-right for free blocks.
    Scalars (size 1) scan right-to-left for free slots.
    No watermarks — pressure equals actual occupied words.
    """

    def __init__(self, scratch_start, scratch_size=SCRATCH_SIZE, scratch_debug=None):
        self.scratch_start = scratch_start
        self.scratch_size = scratch_size
        self.occupied = [False] * scratch_size
        for i in range(scratch_start):
            self.occupied[i] = True
        self.vreg_to_addr = {}
        self.scratch_debug = scratch_debug  # addr -> (name, size) for trace display
        self.peak_usage = 0
        self.peak_cycle = -1
        self.peak_live_snapshot = []
        self.peak_scalars = 0
        self.peak_scalars_cycle = -1

    def try_allocate(self, vreg):
        """Try to allocate space for a vreg. Returns address or None."""
        if not isinstance(vreg, VReg):
            return vreg
        if vreg.pinned_addr is not None:
            return vreg.pinned_addr
        if vreg.parent is not None:
            parent_addr = self.get_addr(vreg.parent)
            return parent_addr + vreg.offset if parent_addr is not None else None
        if vreg in self.vreg_to_addr:
            return self.vreg_to_addr[vreg]
        addr = self._find_space(vreg.size)
        if addr is not None:
            self.vreg_to_addr[vreg] = addr
            if self.scratch_debug is not None:
                self.scratch_debug[addr] = (vreg.name_hint or f"v{vreg.id}", vreg.size)
        return addr

    def _find_space(self, size):
        """Find free space. Vectors scan left, scalars scan right."""
        if size == VLEN:
            for a in range(self.scratch_start, self.scratch_size - VLEN + 1):
                if not any(self.occupied[a:a + VLEN]):
                    for i in range(VLEN):
                        self.occupied[a + i] = True
                    return a
        else:
            for a in range(self.scratch_size - 1, self.scratch_start - 1, -1):
                if not self.occupied[a]:
                    self.occupied[a] = True
                    return a
        return None

    def free(self, vreg):
        """Free a vreg's allocated space."""
        if not isinstance(vreg, VReg) or vreg.pinned_addr is not None:
            return
        if vreg.parent is not None:
            return  # Parent manages the space
        if vreg not in self.vreg_to_addr:
            return
        addr = self.vreg_to_addr[vreg]
        for i in range(vreg.size):
            self.occupied[addr + i] = False
        del self.vreg_to_addr[vreg]

    def rollback(self, vreg):
        """Undo a try_allocate."""
        self.free(vreg)

    def get_addr(self, a):
        """Resolve a VReg or int to a physical address."""
        if isinstance(a, VReg):
            if a.pinned_addr is not None:
                return a.pinned_addr
            if a.parent is not None:
                parent_addr = self.get_addr(a.parent)
                return parent_addr + a.offset if parent_addr is not None else None
            return self.vreg_to_addr.get(a)
        return a

    def rewrite_args(self, args):
        """Rewrite instruction args, replacing VRegs with physical addresses."""
        return tuple(self.get_addr(a) if isinstance(a, (VReg, int)) else a for a in args)

    def current_usage(self):
        """Count currently occupied dynamic words."""
        return sum(self.occupied[self.scratch_start:])

    def track_peak(self, cycle_idx):
        """Snapshot peak usage for diagnostics."""
        usage = self.current_usage()
        if usage > self.peak_usage:
            self.peak_usage = usage
            self.peak_cycle = cycle_idx
            self.peak_live_snapshot = [
                (v, addr) for v, addr in self.vreg_to_addr.items()
            ]
        n_scalars = sum(1 for v in self.vreg_to_addr if v.size == 1)
        if n_scalars > self.peak_scalars:
            self.peak_scalars = n_scalars
            self.peak_scalars_cycle = cycle_idx

    def print_peak_info(self):
        """Print diagnostics about peak scratch usage."""
        budget = self.scratch_size - self.scratch_start
        print(f"Scratch: peak={self.peak_usage}/{budget} at cycle {self.peak_cycle}, peak_scalars={self.peak_scalars} at cycle {self.peak_scalars_cycle}")
        if not self.peak_live_snapshot:
            return
        print(f"  Live vregs at peak ({len(self.peak_live_snapshot)}):")
        by_size = {}
        for v, addr in self.peak_live_snapshot:
            by_size.setdefault(v.size, []).append(v)
        for size, vregs in sorted(by_size.items()):
            words = len(vregs) * size
            print(f"    size={size}: {len(vregs)} vregs ({words} words)")
            prefixes = Counter()
            for v in vregs:
                name = v.name_hint
                prefix = name.rsplit('_v', 1)[0] if '_v' in name else name
                prefix = prefix.rsplit('_r', 1)[0] if '_r' in prefix else prefix
                prefixes[prefix] += 1
            for prefix, count in prefixes.most_common(10):
                print(f"      {prefix}: {count}")


def build_dag(slots):
    """Build a dependency DAG from VReg def-use chains.

    Returns (slot_defs, slot_uses, successors, predecessors, in_degree).
    """
    n = len(slots)
    slot_defs = []
    slot_uses = []
    for engine, args in slots:
        d, u = get_defs_uses(engine, args)
        slot_defs.append(d)
        slot_uses.append(u)

    vreg_definers = defaultdict(list)
    successors = [[] for _ in range(n)]
    predecessors = [[] for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        preds_for_i = set()
        for vreg in slot_uses[i]:
            lookup_vregs = [vreg]
            if hasattr(vreg, 'parent') and vreg.parent is not None:
                lookup_vregs.append(vreg.parent)
            for lv in lookup_vregs:
                for definer in vreg_definers[lv]:
                    if definer not in preds_for_i:
                        preds_for_i.add(definer)
                        successors[definer].append(i)
                        predecessors[i].append(definer)
                        in_degree[i] += 1
        for vreg in slot_defs[i]:
            vreg_definers[vreg].append(i)

    return slot_defs, slot_uses, successors, predecessors, in_degree


def compute_distances(slots, predecessors):
    """Compute BFS distance from each op to its nearest load and flow successor.

    Also tracks which specific load/flow op each op feeds into (target).
    Returns (dist_to_load, dist_to_flow, dist_to_either,
             target_load, target_flow).
    """
    n = len(slots)

    def bfs_dist(target_engines):
        dist = [n] * n
        target = [-1] * n  # which specific bottleneck op this feeds
        queue = deque()
        for i in range(n):
            if slots[i][0] in target_engines:
                dist[i] = 0
                target[i] = i
                queue.append(i)
        while queue:
            j = queue.popleft()
            for pred in predecessors[j]:
                if dist[pred] > dist[j] + 1:
                    dist[pred] = dist[j] + 1
                    target[pred] = target[j]
                    queue.append(pred)
        return dist, target

    dist_to_load, target_load = bfs_dist(("load",))
    dist_to_flow, target_flow = bfs_dist(("flow",))
    dist_to_either = [min(dist_to_load[i], dist_to_flow[i]) for i in range(n)]
    return dist_to_load, dist_to_flow, dist_to_either, target_load, target_flow


def compute_use_counts(slot_uses):
    """Count how many ops use each vreg (resolving parent aliases).

    Returns a dict of {vreg: remaining_use_count} for non-pinned vregs.
    """
    use_count = defaultdict(int)
    for uses in slot_uses:
        for vreg in uses:
            v = vreg.parent if (hasattr(vreg, 'parent') and vreg.parent is not None) else vreg
            if isinstance(v, VReg) and v.pinned_addr is None:
                use_count[v] += 1
    return dict(use_count)


def schedule(slots, slot_limits, allocator, tags=None):
    """Schedule ops using list scheduling with integrated register allocation.

    Builds a DAG from VReg def-use chains and greedily packs independent
    ops into cycles, respecting slot limits. Allocates physical addresses
    inline — no separate allocation pass needed.

    Returns (bundles, sched_meta) where bundles have physical addresses.
    """
    n = len(slots)
    if n == 0:
        return [], []

    slot_defs, slot_uses, successors, predecessors, in_degree = build_dag(slots)
    dist_to_load, dist_to_flow, dist_to_either, target_load, target_flow = compute_distances(slots, predecessors)
    remaining_uses = compute_use_counts(slot_uses)

    # === Active set management ===
    MAX_ACTIVE = 32

    # Extract per-op vector/round info from tags
    op_vi = [-1] * n
    op_rnd = [-1] * n
    ops_per_vi_rnd = defaultdict(list)
    vi_rounds = defaultdict(set)

    if tags:
        for i in range(n):
            tag = tags[i] if i < len(tags) else None
            if tag:
                vi = tag.get('vi', -1)
                rnd = tag.get('rnd', -1)
                op_vi[i] = vi
                op_rnd[i] = rnd
                if vi >= 0:
                    vi_rounds[vi].add(rnd)
                    ops_per_vi_rnd[(vi, rnd)].append(i)

    all_vectors = sorted(vi_rounds.keys())
    has_vectors = len(all_vectors) > 0

    # Yield type: what bottleneck resource does each (vi, rnd) consume?
    round_yield_type = {}
    for (vi, rnd), op_indices in ops_per_vi_rnd.items():
        if rnd < 0:  # setup ops (initial loads)
            round_yield_type[(vi, rnd)] = "free"
            continue
        has_gather = any(slots[i][0] == "load" and slots[i][1][0] == "load_offset"
                        for i in op_indices)
        has_mux = any(slots[i][0] == "flow" and slots[i][1][0] == "vselect"
                      for i in op_indices)
        if has_gather:
            round_yield_type[(vi, rnd)] = "load"
        elif has_mux:
            round_yield_type[(vi, rnd)] = "flow"
        else:
            round_yield_type[(vi, rnd)] = "free"

    # Turnaround cost: serial cycles before this round unlocks VALU work
    # flow ops block 1-per-cycle, loads block at 2-per-cycle
    rnd_turnaround = {}
    for (vi, rnd), op_indices in ops_per_vi_rnd.items():
        if rnd < 0:
            rnd_turnaround[(vi, rnd)] = 0
            continue
        n_flow = sum(1 for i in op_indices if slots[i][0] == "flow")
        n_load = sum(1 for i in op_indices if slots[i][0] == "load")
        rnd_turnaround[(vi, rnd)] = n_flow + (n_load + 1) // 2

    # Active set: first MAX_ACTIVE vectors
    if has_vectors:
        active_set = set(all_vectors[:MAX_ACTIVE])
        inactive_queue = list(all_vectors[MAX_ACTIVE:])
    else:
        active_set = set()
        inactive_queue = []

    # Per-vector round tracking
    total_ops_per_vi_rnd = {k: len(v) for k, v in ops_per_vi_rnd.items()}
    scheduled_per_vi_rnd = defaultdict(int)
    vi_rnd_start_cycle = {}  # (vi, rnd) -> cycle when first op scheduled
    vector_current_round = {}
    for vi in all_vectors:
        vector_current_round[vi] = min(vi_rounds[vi])

    vector_priority = {}  # vi -> priority (0=boost, 1=normal)

    # --- Metadata helpers (for tracing) ---

    def op_desc(i):
        engine, args = slots[i]
        op_name = args[0] if args else ""
        dest = ""
        if len(args) > 1 and isinstance(args[1], VReg):
            dest = args[1].name_hint or f"v{args[1].id}"
        return f"{engine} {op_name} {dest}".strip()

    def named_args(i):
        engine, args = slots[i]
        parts = [a.name_hint or f"v{a.id}" if isinstance(a, VReg) else a for a in args]
        return f"({engine} {' '.join(str(x) for x in parts)})"

    # --- Allocation helpers ---

    def resolve_parent(vreg):
        if hasattr(vreg, 'parent') and vreg.parent is not None:
            return vreg.parent
        return vreg

    def sort_key(i):
        """Prefer ops that free space over ops that consume space."""
        freed = sum(resolve_parent(v).size for v in slot_uses[i]
                    if isinstance(resolve_parent(v), VReg)
                    and resolve_parent(v).pinned_addr is None
                    and remaining_uses.get(resolve_parent(v), 0) == 1)
        added = sum(v.size for v in slot_defs[i]
                    if isinstance(v, VReg) and v.pinned_addr is None
                    and allocator.get_addr(v) is None)
        return added - freed

    def try_alloc_defs(i):
        """Try to allocate all defs for op i. Returns True or rolls back."""
        allocated = []
        for vreg in slot_defs[i]:
            if isinstance(vreg, VReg) and vreg.pinned_addr is None:
                addr = allocator.try_allocate(vreg)
                if addr is None:
                    for v in allocated:
                        allocator.rollback(v)
                    return False
                allocated.append(vreg)
        return True

    transferred_vregs = set()  # vregs whose address was reused by a def

    def try_alloc_or_reuse(i):
        """Try to allocate defs. If that fails, free last-use inputs and retry."""
        if try_alloc_defs(i):
            return True

        # Tentatively free last-use inputs to make space
        pre_freed = []
        for v in slot_uses[i]:
            pv = resolve_parent(v)
            if (isinstance(pv, VReg) and pv.pinned_addr is None
                    and remaining_uses.get(pv, 0) == 1):
                addr = allocator.get_addr(pv)
                if addr is not None:
                    pre_freed.append((pv, addr))
                    for j in range(pv.size):
                        allocator.occupied[addr + j] = False

        if not pre_freed:
            return False

        if try_alloc_defs(i):
            # Mark pre-freed vregs so consume_uses won't double-clear
            for pv, _ in pre_freed:
                transferred_vregs.add(pv.id)
            return True

        # Rollback: restore occupied bits
        for pv, addr in pre_freed:
            for j in range(pv.size):
                allocator.occupied[addr + j] = True
        return False

    def consume_uses(i):
        """Decrement use counts, return vregs ready to free."""
        frees = []
        for v in slot_uses[i]:
            pv = resolve_parent(v)
            if isinstance(pv, VReg) and pv.pinned_addr is None:
                remaining_uses[pv] = remaining_uses.get(pv, 0) - 1
                if remaining_uses[pv] == 0:
                    if pv.id in transferred_vregs:
                        # Address reused by a def — remove mapping, keep occupied
                        if pv in allocator.vreg_to_addr:
                            del allocator.vreg_to_addr[pv]
                        transferred_vregs.discard(pv.id)
                    else:
                        frees.append(pv)
        return frees

    # --- Scheduling state ---

    # Dynamic remaining-predecessors metric: for each bottleneck op (load/flow),
    # count how many unscheduled ops feed into it. As we schedule ops, the count
    # drops, creating momentum toward finishing a group.
    remaining_to_load = defaultdict(int)
    remaining_to_flow = defaultdict(int)
    for i in range(n):
        if target_load[i] >= 0:
            remaining_to_load[target_load[i]] += 1
        if target_flow[i] >= 0:
            remaining_to_flow[target_flow[i]] += 1

    active_metric = "either"  # "load", "flow", or "either"

    def sched_key(i):
        if active_metric == "load":
            d = dist_to_load[i]
            r = remaining_to_load.get(target_load[i], n)
        elif active_metric == "flow":
            d = dist_to_flow[i]
            r = remaining_to_flow.get(target_flow[i], n)
        else:
            d = dist_to_either[i]
            r = min(
                remaining_to_load.get(target_load[i], n),
                remaining_to_flow.get(target_flow[i], n),
            )
        return (d, sort_key(i), i)

    ready = sorted([i for i in range(n) if in_degree[i] == 0], key=sched_key)
    bundles = []
    sched_meta = []
    partial_ops = {}
    stall_slot = 0
    stall_alloc = 0

    ready_cycle = {}
    op_scheduled_cycle = {}
    op_to_id = {}
    next_op_id = [0]
    cycle_num = 0
    for i in ready:
        ready_cycle[i] = 0

    def assign_op_id(i):
        if i not in op_to_id:
            op_to_id[i] = next_op_id[0]
            next_op_id[0] += 1
        return op_to_id[i]

    def build_meta(i):
        meta = {
            "ready": ready_cycle.get(i, 0), "sched": cycle_num,
            "op_id": assign_op_id(i), "named": named_args(i),
            "pressure": sort_key(i),
            "dist_to_load": dist_to_load[i], "dist_to_flow": dist_to_flow[i],
            "deps": [{"op": op_desc(p), "cycle": op_scheduled_cycle.get(p, -1),
                       "op_id": assign_op_id(p)} for p in predecessors[i]],
        }
        if tags is not None and i < len(tags):
            meta.update(tags[i])
        return meta

    # --- Main scheduling loop ---

    while ready or partial_ops:
        bundle = []
        bundle_meta = []
        available = dict(slot_limits)
        scheduled_this_cycle = []
        remaining = []
        pending_frees = []

        def mark_scheduled(i):
            """Update dynamic metrics and bookkeeping when op i completes."""
            scheduled_this_cycle.append(i)
            op_scheduled_cycle[i] = cycle_num
            pending_frees.extend(consume_uses(i))
            if target_load[i] >= 0:
                remaining_to_load[target_load[i]] -= 1
            if target_flow[i] >= 0:
                remaining_to_flow[target_flow[i]] -= 1

        def emit(slot_tuple, i):
            """Append a slot to the bundle and mark op i as scheduled."""
            bundle.append(slot_tuple)
            bundle_meta.append(build_meta(i))
            mark_scheduled(i)

        # Re-admit vectors to active set based on scheduling priority
        if has_vectors and len(active_set) < MAX_ACTIVE:
            # Find inactive vectors that have gated ready ops
            candidate_vis = {}
            for i in ready:
                vi_i = op_vi[i]
                if vi_i >= 0 and vi_i not in active_set:
                    if vi_i not in candidate_vis or sched_key(i) < candidate_vis[vi_i]:
                        candidate_vis[vi_i] = sched_key(i)
            # Admit best candidates up to MAX_ACTIVE
            if candidate_vis:
                ranked = sorted(candidate_vis.keys(), key=lambda vi: candidate_vis[vi])
                slots_avail = MAX_ACTIVE - len(active_set)
                for vi in ranked[:slots_avail]:
                    active_set.add(vi)

        # Continue in-progress partial valu→alu promotions
        for i in list(partial_ops):
            alu_avail = available.get("alu", 0)
            if alu_avail <= 0:
                break
            done_so_far = partial_ops[i]
            can_do = min(alu_avail, VLEN - done_so_far)
            physical_args = allocator.rewrite_args(slots[i][1])
            bundle.append(("valu_as_alu", physical_args, done_so_far, can_do))
            bundle_meta.append(build_meta(i))
            available["alu"] -= can_do
            partial_ops[i] = done_so_far + can_do
            if partial_ops[i] >= VLEN:
                del partial_ops[i]
                mark_scheduled(i)

        # Compute vector priorities based on turnaround cost to VALU work
        if has_vectors:
            n_ready_valu = sum(1 for i in ready if slots[i][0] == "valu")
            alu_pressure = n_ready_valu > slot_limits.get("valu", 6)

            vector_priority.clear()
            if alu_pressure:
                # ALU promotion under pressure — boost vectors with lowest
                # turnaround to new VALU work (low-k rounds finish faster)
                for vi in active_set:
                    crnd = vector_current_round.get(vi)
                    if crnd is None:
                        vector_priority[vi] = 99
                    else:
                        vector_priority[vi] = rnd_turnaround.get((vi, crnd), 0)
            else:
                for vi in active_set:
                    vector_priority[vi] = 0

        # Adaptive starvation: bias remaining-preds metric toward starved resource
        n_ready_loads = sum(1 for i in ready if slots[i][0] == "load")
        n_ready_flows = sum(1 for i in ready if slots[i][0] == "flow")
        load_starved = n_ready_loads < slot_limits.get("load", 2)
        flow_starved = n_ready_flows < slot_limits.get("flow", 1)
        if load_starved and not flow_starved:
            active_metric = "load"
        elif flow_starved and not load_starved:
            active_metric = "flow"
        else:
            active_metric = "either"

        ready.sort(key=sched_key)

        # Schedule ready ops
        for i in ready:
            engine = slots[i][0]

            # Active set gating: gate ALL ops for inactive vectors
            vi_i = op_vi[i]
            if vi_i >= 0 and vi_i not in active_set:
                remaining.append(i)
                continue

            if engine == "flex_alu_add":
                # flex_alu_add: scalar add emitted as alu, args trimmed to drop imm
                if available.get("alu", 0) <= 0:
                    remaining.append(i)
                    stall_slot += 1
                    continue
                if not try_alloc_or_reuse(i):
                    remaining.append(i)
                    stall_alloc += 1
                    continue
                physical_args = allocator.rewrite_args(slots[i][1])
                emit(("alu", physical_args[:4]), i)
                available["alu"] -= 1
                continue

            # Check engine availability
            can_native = available.get(engine, 0) > 0
            can_promote = (engine == "valu"
                          and slots[i][1][0] not in ("vbroadcast", "multiply_add")
                          and available.get("alu", 0) > 0)

            if not can_native and not can_promote:
                remaining.append(i)
                stall_slot += 1
                continue

            if not try_alloc_or_reuse(i):
                remaining.append(i)
                stall_alloc += 1
                if stall_alloc <= 5:
                    defs_info = [(v.name_hint, v.size) for v in slot_defs[i] if isinstance(v, VReg) and v.pinned_addr is None]
                    print(f"  alloc_fail cycle={cycle_num} op={op_desc(i)} usage={allocator.current_usage()} defs={defs_info}")
                continue

            if can_native:
                physical_args = allocator.rewrite_args(slots[i][1])
                emit((engine, physical_args), i)
                available[engine] -= 1
            else:
                # valu→alu promotion (may be partial if not enough alu slots)
                alu_avail = available.get("alu", 0)
                can_do = min(alu_avail, VLEN)
                physical_args = allocator.rewrite_args(slots[i][1])
                bundle.append(("valu_as_alu", physical_args, 0, can_do))
                bundle_meta.append(build_meta(i))
                available["alu"] -= can_do
                op_scheduled_cycle[i] = cycle_num
                if can_do >= VLEN:
                    mark_scheduled(i)
                else:
                    partial_ops[i] = can_do

        allocator.track_peak(cycle_num)

        for vreg in pending_frees:
            allocator.free(vreg)

        # Unblock successors of completed ops
        newly_ready = []
        for i in scheduled_this_cycle:
            for succ in successors[i]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    newly_ready.append(succ)
                    ready_cycle[succ] = cycle_num + 1

        # Track per-vector round completion and active set transitions
        if has_vectors:
            for i in scheduled_this_cycle:
                vi_s = op_vi[i]
                rnd_s = op_rnd[i]
                if vi_s < 0:
                    continue
                key = (vi_s, rnd_s)
                if key in total_ops_per_vi_rnd:
                    if key not in vi_rnd_start_cycle:
                        vi_rnd_start_cycle[key] = cycle_num
                    scheduled_per_vi_rnd[key] += 1
                    if scheduled_per_vi_rnd[key] >= total_ops_per_vi_rnd[key]:
                        start_c = vi_rnd_start_cycle.get(key, -1)
                        print(f"  (vi={vi_s}, rnd={rnd_s}) done cycle={cycle_num} (started={start_c}, dur={cycle_num - start_c})")
                        # (vi, rnd) complete — advance vector to next round
                        next_rounds = sorted(
                            r for r in vi_rounds[vi_s]
                            if scheduled_per_vi_rnd[(vi_s, r)] < total_ops_per_vi_rnd.get((vi_s, r), 0)
                        )
                        # Round complete — remove from active set
                        active_set.discard(vi_s)
                        if next_rounds:
                            vector_current_round[vi_s] = next_rounds[0]

        ready = sorted(remaining + newly_ready, key=sched_key)
        if bundle:
            bundles.append(bundle)
            sched_meta.append(bundle_meta)
            cycle_num += 1
        elif not partial_ops:
            if ready:
                gated = [i for i in ready if op_vi[i] >= 0 and op_vi[i] not in active_set]
                not_gated = [i for i in ready if not (op_vi[i] >= 0 and op_vi[i] not in active_set)]
                engines = Counter(slots[i][0] for i in ready)
                print(f"  DEADLOCK cycle={cycle_num}: {len(ready)} ready, {len(gated)} gated, {len(not_gated)} not gated")
                print(f"  Engines: {dict(engines)}, Scratch: {allocator.current_usage()}/{allocator.scratch_size}")
                print(f"  Active set ({len(active_set)}): {sorted(active_set)}")
                for i in not_gated[:10]:
                    defs = [(v.name_hint, v.size) for v in slot_defs[i] if isinstance(v, VReg)]
                    print(f"    op {i}: {op_desc(i)} vi={op_vi[i]} rnd={op_rnd[i]} engine={slots[i][0]} defs={defs}")
                raise RuntimeError(
                    f"Scheduler deadlock: {len(ready)} ready ops but none schedulable."
                )
            break

    print(f"Scheduler: {cycle_num} cycles, {n} ops, stalls: slot_full={stall_slot}, alloc_fail={stall_alloc}")
    if has_vectors:
        print(f"  Active set: max={MAX_ACTIVE}")
        print(f"  Vectors: {len(all_vectors)} total, {len(inactive_queue)} remaining inactive")
    return bundles, sched_meta


def expand_valu_as_alu(bundles, sched_meta=None):
    """Expand valu_as_alu slots into scalar alu ops.

    Each valu_as_alu slot is a 4-tuple: (engine, args, start, count)
    specifying which element range to expand.
    Must be called AFTER register allocation (physical addresses assigned).
    """
    result = []
    result_meta = [] if sched_meta is not None else None
    for bi, bundle in enumerate(bundles):
        new_bundle = []
        new_meta = [] if sched_meta is not None else None
        for si, slot in enumerate(bundle):
            meta = sched_meta[bi][si] if sched_meta is not None else None
            if slot[0] == "valu_as_alu":
                args, start, count = slot[1], slot[2], slot[3]
                op = args[0]
                dest = args[1]
                sources = args[2:]
                for i in range(start, start + count):
                    new_args = (op, dest + i) + tuple(s + i for s in sources)
                    new_bundle.append(("alu", new_args))
                    if new_meta is not None:
                        new_meta.append(meta)
            else:
                new_bundle.append((slot[0], slot[1]))
                if new_meta is not None:
                    new_meta.append(meta)
        result.append(new_bundle)
        if result_meta is not None:
            result_meta.append(new_meta)
    return result, result_meta


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vregs = {}  # name -> VReg for named vregs
        self.pending_const_loads = []  # ("const", addr, val) tuples to batch-emit
        self.pending_mem_loads = []    # ("load", dest, src) tuples to batch-emit

    @staticmethod
    def _extract_timing(sched_meta):
        """Extract per-(vi, rnd) and per-op timing from sched_meta."""
        timing = {}  # (vi, rnd) -> {"start", "end", "ops"}
        op_stats = []  # per-op: {"cycle", "ready", "wait", "engine", "vi", "rnd", ...}

        for bi, bundle_meta in enumerate(sched_meta):
            for meta in bundle_meta:
                if meta is None:
                    continue
                vi = meta.get("vi", -1)
                rnd = meta.get("rnd", -1)
                cycle = meta.get("sched", 0)
                ready = meta.get("ready", 0)

                # Per-op stats
                op_stats.append({
                    "cycle": cycle, "ready": ready, "wait": cycle - ready,
                    "vi": vi, "rnd": rnd,
                    "dist_to_load": meta.get("dist_to_load", -1),
                    "dist_to_flow": meta.get("dist_to_flow", -1),
                    "pressure": meta.get("pressure", 0),
                    "named": meta.get("named", ""),
                })

                if vi < 0 or rnd < 0:
                    continue
                key = (vi, rnd)
                if key not in timing:
                    timing[key] = {"start": cycle, "end": cycle, "ops": 0}
                timing[key]["start"] = min(timing[key]["start"], cycle)
                timing[key]["end"] = max(timing[key]["end"], cycle)
                timing[key]["ops"] += 1

        return timing, op_stats

    def print_timing_summary(self):
        """Print comprehensive scheduling analysis."""
        if not hasattr(self, 'vi_rnd_timing') or not self.vi_rnd_timing:
            print("No timing data available.")
            return
        import pandas as pd
        timing, op_stats = self.vi_rnd_timing, self._op_stats

        # === 1. Per-round duration stats ===
        rows = []
        for (vi, rnd), t in timing.items():
            rows.append({"vi": vi, "rnd": rnd, "start": t["start"],
                         "end": t["end"], "duration": t["end"] - t["start"],
                         "ops": t["ops"]})
        df = pd.DataFrame(rows).sort_values(["rnd", "vi"])

        print("\n=== Per-round duration stats ===")
        grouped = df.groupby("rnd")
        summary = pd.DataFrame({
            "mean_dur": grouped["duration"].mean().round(1),
            "min_dur": grouped["duration"].min(),
            "max_dur": grouped["duration"].max(),
            "spread": (grouped["duration"].max() - grouped["duration"].min()),
            "first_start": grouped["start"].min(),
            "last_end": grouped["end"].max(),
            "wall_time": grouped["end"].max() - grouped["start"].min(),
        })
        print(summary.to_string())

        # === 2. Round overlap: how many rounds are in-flight per cycle ===
        total_cycles = df["end"].max() + 1
        rounds_active = [0] * total_cycles
        for _, row in df.iterrows():
            for c in range(row["start"], row["end"] + 1):
                rounds_active[c] = max(rounds_active[c], 1)  # just mark active
        # Count distinct rounds active per cycle
        rnd_per_cycle = defaultdict(set)
        for _, row in df.iterrows():
            for c in range(row["start"], row["end"] + 1):
                rnd_per_cycle[c].add(row["rnd"])
        overlap_counts = [len(rnd_per_cycle[c]) for c in range(total_cycles)]
        if overlap_counts:
            print(f"\n=== Round overlap ===")
            print(f"  Max rounds in-flight: {max(overlap_counts)}")
            avg_overlap = sum(overlap_counts) / len(overlap_counts)
            print(f"  Avg rounds in-flight: {avg_overlap:.1f}")
            # Drain phase: cycles where overlap drops to 1
            drain_start = total_cycles
            for c in range(total_cycles - 1, -1, -1):
                if overlap_counts[c] > 1:
                    drain_start = c + 1
                    break
            print(f"  Drain phase starts: cycle {drain_start} ({total_cycles - drain_start} drain cycles)")

        # === 3. Op wait times (ready → scheduled delay) ===
        ops_df = pd.DataFrame(op_stats)
        if len(ops_df) > 0 and "wait" in ops_df.columns:
            print(f"\n=== Op wait times (ready → scheduled) ===")
            print(f"  Mean: {ops_df['wait'].mean():.1f}, Median: {ops_df['wait'].median():.0f}, "
                  f"Max: {ops_df['wait'].max()}, P95: {ops_df['wait'].quantile(0.95):.0f}")
            # Wait time by round
            rnd_waits = ops_df[ops_df["rnd"] >= 0].groupby("rnd")["wait"]
            if len(rnd_waits) > 0:
                print(f"  Per-round mean wait: min={rnd_waits.mean().min():.1f}, "
                      f"max={rnd_waits.mean().max():.1f}")

        # === 4. Engine utilization from bundles ===
        if hasattr(self, '_bundles_for_stats'):
            bundles = self._bundles_for_stats
            engine_counts = defaultdict(int)
            for bundle in bundles:
                for slot in bundle:
                    eng = slot[0]
                    if eng == "valu_as_alu":
                        engine_counts["alu"] += slot[3]  # count = number of scalar ops
                    else:
                        engine_counts[eng] += 1
            n_cycles = len(bundles)
            print(f"\n=== Engine utilization ({n_cycles} cycles) ===")
            for eng in ["valu", "alu", "load", "store", "flow"]:
                count = engine_counts.get(eng, 0)
                limit = SLOT_LIMITS.get(eng, 0)
                util = count / (n_cycles * limit) * 100 if limit > 0 and n_cycles > 0 else 0
                print(f"  {eng:6s}: {count:6d} slots used, {util:5.1f}% utilization "
                      f"({count/n_cycles:.1f}/{limit} per cycle)")

        print(f"\n=== Overall ===")
        print(f"  Mean duration: {df['duration'].mean():.1f} cycles")
        print(f"  Total (vi,rnd) pairs: {len(df)}")
        return df

    def plot_timing(self):
        """Plot histograms of per-(vi, rnd) durations and a Gantt-like chart."""
        df = self.print_timing_summary()
        if df is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram of durations
        axes[0].hist(df["duration"], bins=30, edgecolor="black")
        axes[0].set_xlabel("Duration (cycles)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Duration distribution (all vi×rnd)")

        # Gantt chart: each row is a (vi, rnd), colored by round
        df_sorted = df.sort_values(["rnd", "vi"])
        colors = plt.cm.tab20(df_sorted["rnd"] % 20)
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            axes[1].barh(idx, row["duration"], left=row["start"],
                        color=colors[idx], height=0.8)
        axes[1].set_xlabel("Cycle")
        axes[1].set_ylabel("(vi, rnd) index")
        axes[1].set_title("Gantt: start→end per (vi, rnd)")

        # Per-round box plot of durations
        rounds = sorted(df["rnd"].unique())
        data = [df[df["rnd"] == r]["duration"].values for r in rounds]
        axes[2].boxplot(data, labels=[str(r) for r in rounds])
        axes[2].set_xlabel("Round")
        axes[2].set_ylabel("Duration (cycles)")
        axes[2].set_title("Duration spread per round")

        plt.tight_layout()
        plt.savefig("timing_analysis.png", dpi=150)
        plt.show()
        print("Saved timing_analysis.png")

        # === Per-round timeline: one subplot per round, vi on y-axis ===
        rounds = sorted(df["rnd"].unique())
        n_rounds = len(rounds)
        ncols = 4
        nrows = (n_rounds + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                                    sharex=True, sharey=True)
        axes2 = axes2.flatten() if n_rounds > 1 else [axes2]
        global_start = df["start"].min()
        global_end = df["end"].max()
        cmap = plt.cm.tab10
        for idx, rnd in enumerate(rounds):
            ax = axes2[idx]
            rnd_df = df[df["rnd"] == rnd].sort_values("vi")
            for _, row in rnd_df.iterrows():
                ax.barh(row["vi"], row["duration"], left=row["start"],
                        color=cmap(rnd % 10), edgecolor="black", linewidth=0.5,
                        height=0.8)
            ax.set_title(f"Round {rnd}")
            ax.set_xlim(global_start, global_end)
            if idx % ncols == 0:
                ax.set_ylabel("vi")
            if idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Cycle")
        # Hide unused subplots
        for idx in range(n_rounds, len(axes2)):
            axes2[idx].set_visible(False)
        fig2.suptitle("Per-round timeline: start→end per vector", fontsize=14)
        fig2.tight_layout()
        fig2.savefig("per_round_timeline.png", dpi=150)
        plt.show()
        print("Saved per_round_timeline.png")

    def debug_info(self):
        return DebugInfo(
            scratch_map=self.scratch_debug,
            slot_sched_info=getattr(self, 'slot_sched_info', None),
        )

    def new_vreg(self, name_hint="", size=1):
        """Create a new unique virtual register."""
        return VReg(name_hint=name_hint, size=size)

    def new_vreg_vec(self, name_hint=""):
        """Create a new unique vector virtual register."""
        return VReg(name_hint=name_hint, size=VLEN)

    def sub_vreg(self, parent, offset):
        """Create a scalar vreg aliased to parent vector vreg at given offset.
        Resolves to parent's physical address + offset during allocation."""
        child = VReg(name_hint=f"{parent.name_hint}[{offset}]", size=1)
        child.parent = parent
        child.offset = offset
        return child

    def pinned_vreg(self, name, size=1):
        """Create or get a named virtual register (deduped by name)."""
        if name not in self.vregs:
            vreg = VReg(name_hint=name, size=size)
            self.vregs[name] = vreg
        return self.vregs[name]

    def build(self, bundles: list[list[tuple[Engine, tuple]]], sched_meta=None):
        """
        Convert bundles of slots into instruction format.

        Args:
            bundles: List of bundles, where each bundle is a list of (engine, args) slots.
            sched_meta: Optional parallel structure with scheduling metadata per slot.

        Returns:
            (instrs, slot_sched_info) where slot_sched_info maps
            (pc, engine, slot_index_within_engine) -> metadata dict.
        """
        instrs = []
        slot_sched_info = {}
        for bi, bundle in enumerate(bundles):
            instr = defaultdict(list)
            engine_count = defaultdict(int)
            for si, (engine, args) in enumerate(bundle):
                slot_idx = engine_count[engine]
                engine_count[engine] += 1
                instr[engine].append(args)
                if sched_meta is not None and sched_meta[bi][si] is not None:
                    slot_sched_info[(bi, engine, slot_idx)] = sched_meta[bi][si]
            instrs.append(dict(instr))
        return instrs, slot_sched_info

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        if self.scratch_ptr > SCRATCH_SIZE:
            assert False, (f"Out of scratch: ptr={self.scratch_ptr}/{SCRATCH_SIZE}, "
                          f"allocating '{name}' (size={length})")
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            vreg_name = name or f"const_{val}"
            vreg = self.new_vreg(vreg_name)
            self.pending_const_loads.append(("load", ("const", vreg, val)))
            self.const_map[val] = vreg
        return self.const_map[val]

    def build_vhash(self, val_in, hash_const_vecs):
        """Vectorized hash - operates on VLEN elements at once.
        Uses virtual registers. Returns (slots, val_out) where val_out is the result vreg."""
        slots = []
        val_vec = val_in

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec = hash_const_vecs[hi * 2]
            const3_vec = hash_const_vecs[hi * 2 + 1]

            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Fuse: (a + const) + (a << N) = a * (2^N + 1) + const
                val_out = self.new_vreg_vec(f"hash{hi}_out")
                slots.append(("valu", ("multiply_add", val_out, val_vec, const3_vec, const1_vec)))
            else:
                # General case: 3 ops
                tmp1 = self.new_vreg_vec(f"hash{hi}_t1")
                tmp2 = self.new_vreg_vec(f"hash{hi}_t2")
                val_out = self.new_vreg_vec(f"hash{hi}_out")
                slots.append(("valu", (op1, tmp1, val_vec, const1_vec)))
                slots.append(("valu", (op3, tmp2, val_vec, const3_vec)))
                slots.append(("valu", (op2, val_out, tmp1, tmp2)))

            val_vec = val_out  # Chain to next stage

        return slots, val_vec

    def preload_level(self, k, forest_values_p_addr):
        """Preload all 2^k node values at tree level k into broadcast vectors.

        Level k has 2^k nodes starting at tree index (2^k - 1).
        Loads them via vload (8 words at a time), then vbroadcasts each
        scalar into its own vector vreg for use by build_mux_select.

        Returns (slots, broadcast_vregs).
        """
        slots = []
        n_nodes = 2 ** k

        level_offset_imm = 2**k - 1
        level_offset = self.scratch_const(level_offset_imm)
        level_base = self.new_vreg(f"level{k}_base")
        slots.append(("flex_alu_add", ("+", level_base, forest_values_p_addr, level_offset, level_offset_imm)))
        vregs = []

        for i in range(max(n_nodes//8, 1)):
            current_offset = self.new_vreg(f"level{k}_offset_{i}")
            offset_imm = i * VLEN
            offset_constant = self.scratch_const(offset_imm)
            slots.append(("flex_alu_add", ("+", current_offset, level_base, offset_constant, offset_imm)))
            vnode_vreg = self.new_vreg_vec(f"level{k}_offset_node_{i}")
            slots.append(("load", ("vload", vnode_vreg, current_offset)))
            vregs.append(vnode_vreg)

        broadcast_vregs = []
        for i, vreg in enumerate(vregs):
            for j in range(min(n_nodes,8)):
                bcast = self.new_vreg_vec(f"level{k}_bcast{i*8 + j}")
                broadcast_vregs.append(bcast)
                slots.append(("valu", ("vbroadcast", bcast, self.sub_vreg(vreg, j))))

        return slots, broadcast_vregs


    def build_mux_select(self, broadcast_vregs, idx_vreg, k, index):
        """Select each element's node value from preloaded broadcast vectors using a mux tree.

        Extracts position bits from idx_vreg and uses k stages of vselect
        to narrow 2^k candidates down to 1 per element.

        Returns (slots, result_vreg).
        """
        # Mask vecs for bit extraction: bit s -> AND with (1 << s)
        # vselect checks != 0, so we don't need to shift the bit down to position 0
        _mask_vecs = {0: "one_vec", 1: "two_vec", 2: "four_vec", 3: "eight_vec"}

        def extract_bit(idx_vreg, num_bits, k, i):
            # returns (slots, bit_vreg) where bit_vreg is nonzero iff bit num_bits is set
            s = []
            mask_name = _mask_vecs.get(num_bits)
            if mask_name:
                mask_vec = self.pinned_vreg(mask_name, VLEN)
            else:
                # For higher bits (k>=3), would need a new broadcast constant
                mask_vec = self.pinned_vreg(f"mask_{1 << num_bits}_vec", VLEN)
            bit = self.new_vreg_vec(f"bit_{num_bits}_stage{k}_vec{i}")
            s.append(("valu", ("&", bit, idx_vreg, mask_vec)))
            return s, bit
        
        slots = []
        remaining_vregs = list(broadcast_vregs)
        if k == 0:
            return slots, broadcast_vregs[0]

        if k == 1:
            # 1-based: idx' is 2 or 3 at k=1. idx'&1 = 0 for left (node 2), 1 for right (node 3).
            bit = self.new_vreg_vec(f"bit_0_stage0_vec{index}")
            slots.append(("valu", ("&", bit, idx_vreg, self.pinned_vreg("one_vec", VLEN))))
            result = self.new_vreg_vec(f"mux_stage0_vec{index}_0")
            # Non-inverted: bit=0 selects broadcast[0] (left), bit=1 selects broadcast[1] (right)
            slots.append(("flow", ("vselect", result, bit, broadcast_vregs[1], broadcast_vregs[0])))
            return slots, result

        # 1-based: level k starts at 2^k, so position = low k bits of idx'.
        # No subtract needed — bit extraction only looks at low bits.
        # Subtract kept: removing it saves ops but hurts scheduling (fewer deps = more
        # scratch pressure). With subtract: 1231 cycles, without: 1266 cycles.
        _level_start_1based = {2: "four_vec", 3: "eight_vec", 4: "sixteen_vec"}
        adjusted_idx = self.new_vreg_vec(f"mux{k}_adjusted_idx_vec{index}")
        slots.append(("valu", ("-", adjusted_idx, idx_vreg, self.pinned_vreg(_level_start_1based[k], VLEN))))
        bit_source = adjusted_idx

        for stage in range(k):
            shift_slots, condition_vreg = extract_bit(bit_source, stage, stage, index)
            slots.extend(shift_slots)
            next_vregs = []
            for i in range(max(len(remaining_vregs)//2, 1)):
                # grab i and i + 1, vselect between them  
                next_vreg = remaining_vregs.pop(0)
                adjacent_vreg = remaining_vregs.pop(0)
                result_vreg = self.new_vreg_vec(f"mux_stage{stage}_vec{index}_{i}")
                slots.append(("flow", ("vselect", result_vreg, condition_vreg, adjacent_vreg, next_vreg)))  
                next_vregs.append(result_vreg)
            remaining_vregs = next_vregs

        return slots, remaining_vregs[0]

    def build_gather(self, addr_vec, name_hint="gather"):
        """Gather VLEN values from non-contiguous addresses into a vector.
        Returns (slots, dest_vreg) where dest_vreg is the result."""
        dest_vec = self.new_vreg_vec(name_hint)
        slots = []
        for i in range(VLEN):
            slots.append(("load", ("load_offset", dest_vec, addr_vec, i)))
        return slots, dest_vec

    def setup_kernel_scratch_and_constants(self):
        """Allocate scratch space, load kernel parameters, and register constants.

        Returns (zero_const, one_const, two_const, param_vregs).
        """
        # Scratch space addresses for kernel parameters
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        param_vregs = {}
        for v in init_vars:
            param_vregs[v] = self.pinned_vreg(v)

        # Use a separate temp per param so all pairs are independent
        for i, v in enumerate(init_vars):
            tmp = self.new_vreg(f"param_idx_{i}")
            self.pending_const_loads.append(("load", ("const", tmp, i)))
            self.pending_mem_loads.append(("load", ("load", param_vregs[v], tmp)))

        # Pre-load basic constants and hash constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        for _, val1, _, _, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)

        return zero_const, one_const, two_const, param_vregs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
        slot_limits=None,
    ):
        """
        Vectorized kernel using virtual registers.
        Each write creates a fresh vreg (SSA form).
        """
        # Setup phase: allocate scratch and register constants
        zero_const, one_const, two_const, param_vregs = self.setup_kernel_scratch_and_constants()

        # Emit pause immediately (no-op sync point for reference_kernel2)
        self.add("flow", ("pause",))

        body = []  # array of (engine, args) slots with virtual registers
        body_tags = []  # parallel metadata: {"vi": ..., "rnd": ...} per slot

        def emit(slot, vi=-1, rnd=-1):
            body.append(slot)
            body_tags.append({"vi": vi, "rnd": rnd})

        def emit_all(slots, vi=-1, rnd=-1):
            for s in slots:
                emit(s, vi, rnd)

        # Pinned vregs for broadcast constants (allocated once, used throughout)
        zero_vec = self.pinned_vreg("zero_vec", VLEN)
        one_vec = self.pinned_vreg("one_vec", VLEN)
        two_vec = self.pinned_vreg("two_vec", VLEN)
        four_vec = self.pinned_vreg("four_vec", VLEN)
        eight_vec = self.pinned_vreg("eight_vec", VLEN)
        sixteen_vec = self.pinned_vreg("sixteen_vec", VLEN)
        forest_p_vec = self.pinned_vreg("forest_p_vec", VLEN)

        emit(("valu", ("vbroadcast", zero_vec, zero_const)))
        emit(("valu", ("vbroadcast", one_vec, one_const)))
        emit(("valu", ("vbroadcast", two_vec, two_const)))
        emit(("valu", ("vbroadcast", four_vec, self.scratch_const(4))))
        emit(("valu", ("vbroadcast", eight_vec, self.scratch_const(8))))
        emit(("valu", ("vbroadcast", sixteen_vec, self.scratch_const(16))))
        emit(("valu", ("vbroadcast", forest_p_vec, param_vregs["forest_values_p"])))
        # 1-based indexing: gather uses forest_p - 1 + idx' instead of forest_p + idx
        forest_p_m1_vec = self.pinned_vreg("forest_p_m1_vec", VLEN)
        minus_one_const = self.scratch_const(0xFFFFFFFF)  # -1 mod 2^32
        emit(("valu", ("vbroadcast", forest_p_m1_vec, minus_one_const)))
        # forest_p_m1_vec = forest_p_vec + (-1) = forest_p - 1
        forest_p_m1_final = self.pinned_vreg("forest_p_m1_final", VLEN)
        emit(("valu", ("+", forest_p_m1_final, forest_p_vec, forest_p_m1_vec)))
        # 1-based indexing: level start constants are powers of 2, already broadcast
        # as four_vec (k=2) and eight_vec (k=3). No extra broadcasts needed.

        # Pre-broadcast all 12 hash constants (pinned since used every iteration)
        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec = self.pinned_vreg(f"hash_c1_{hi}_vec", VLEN)
            const3_vec = self.pinned_vreg(f"hash_c3_{hi}_vec", VLEN)
            emit(("valu", ("vbroadcast", const1_vec, self.scratch_const(val1))))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # For fusable stages, const3 becomes the multiplier 2^N + 1
                emit(("valu", ("vbroadcast", const3_vec, self.scratch_const((1 << val3) + 1))))
            else:
                emit(("valu", ("vbroadcast", const3_vec, self.scratch_const(val3))))
            hash_const_vecs.append(const1_vec)
            hash_const_vecs.append(const3_vec)

        n_vectors = batch_size // VLEN

        # Pre-register offset constants so they're in the batch
        for vi in range(n_vectors):
            self.scratch_const(vi * VLEN)

        # Load indices and values from memory into scratch (once)
        idx_vecs = []  # current index vector per vi chunk
        val_vecs = []  # current value vector per vi chunk
        for vi in range(n_vectors):
            offset_imm = vi * VLEN
            offset_const = self.scratch_const(offset_imm)
            idx_base = self.new_vreg(f"idx_base_init_v{vi}")
            val_base = self.new_vreg(f"val_base_init_v{vi}")
            emit(("flex_alu_add", ("+", idx_base, param_vregs["inp_indices_p"], offset_const, offset_imm)), vi=vi, rnd=-1)
            emit(("flex_alu_add", ("+", val_base, param_vregs["inp_values_p"], offset_const, offset_imm)), vi=vi, rnd=-1)

            idx_v_raw = self.new_vreg_vec(f"idx_raw_v{vi}")
            val_v = self.new_vreg_vec(f"val_init_v{vi}")
            emit(("load", ("vload", idx_v_raw, idx_base)), vi=vi, rnd=-1)
            emit(("load", ("vload", val_v, val_base)), vi=vi, rnd=-1)
            # Convert to 1-based indexing: idx' = idx + 1
            idx_v = self.new_vreg_vec(f"idx_init_v{vi}")
            emit(("valu", ("+", idx_v, idx_v_raw, one_vec)), vi=vi, rnd=-1)
            idx_vecs.append(idx_v)
            val_vecs.append(val_v)

        # Main loop: all rounds, reading/writing scratch VRegs (no memory round-trip)
        cached_broadcasts = {}
        for rnd in range(rounds):
            new_idx_vecs = []
            new_val_vecs = []
            k = rnd % (forest_height + 1)
            # Mux/gather sets up node values for this round's hash.
            # Tag as prev flight (rnd-1) so it's part of the previous flight.
            setup_rnd = rnd - 1 if rnd > 0 else -1

            if k <= 3:
                if k in cached_broadcasts:
                    broadcast_vregs = cached_broadcasts[k]
                else:
                    preload_slots, broadcast_vregs = self.preload_level(k, param_vregs["forest_values_p"])
                    emit_all(preload_slots, rnd=setup_rnd)
                    cached_broadcasts[k] = broadcast_vregs

            # Optimal mux/gather split: balance flow (mux) vs load (gather)
            # m = 4n / (2^k + 3), rounded to nearest int
            mux_count = {0: n_vectors, 1: n_vectors, 2: n_vectors, 3: n_vectors - 2, 4: 1}

            for vi in range(n_vectors):
                idx_loaded = idx_vecs[vi]
                val_loaded = val_vecs[vi]

                # Compute gather addresses: addr = forest_p + idx
                if k <= 3 and vi < mux_count[k]:
                    select_slots, node_val = self.build_mux_select(broadcast_vregs, idx_loaded, k, vi)
                    emit_all(select_slots, vi=vi, rnd=setup_rnd)

                else:
                    addr_vec = self.new_vreg_vec(f"addr_r{rnd}_v{vi}")
                    # 1-based: addr = (forest_p - 1) + idx'
                    emit(("valu", ("+", addr_vec, forest_p_m1_final, idx_loaded)), vi=vi, rnd=setup_rnd)

                    # Gather node values from tree (still from main memory)
                    gather_slots, node_val = self.build_gather(addr_vec, f"node_r{rnd}_v{vi}")
                    emit_all(gather_slots, vi=vi, rnd=setup_rnd)

                # val = val ^ node_val
                val_xored = self.new_vreg_vec(f"xor_r{rnd}_v{vi}")
                emit(("valu", ("^", val_xored, val_loaded, node_val)), vi=vi, rnd=rnd)

                # val = myhash(val)
                hash_slots, val_hashed = self.build_vhash(val_xored, hash_const_vecs)
                emit_all(hash_slots, vi=vi, rnd=rnd)

                if k == forest_height:
                    # At leaves: idx wraps to root. 1-based root = 1
                    new_idx_vecs.append(one_vec)
                else:
                    # 1-based: idx_next' = 2*idx' + parity (2 ops instead of 3)
                    parity = self.new_vreg_vec(f"parity_r{rnd}_v{vi}")
                    idx_next = self.new_vreg_vec(f"idx_next_r{rnd}_v{vi}")

                    emit(("valu", ("&", parity, val_hashed, one_vec)), vi=vi, rnd=rnd)
                    emit(("valu", ("multiply_add", idx_next, idx_loaded, two_vec, parity)), vi=vi, rnd=rnd)
                    new_idx_vecs.append(idx_next)

                new_val_vecs.append(val_hashed)

            idx_vecs = new_idx_vecs
            val_vecs = new_val_vecs

        # Store final results back to memory (once)
        # Convert idx back from 1-based to 0-based: idx = idx' - 1 = idx' + 0xFFFFFFFF
        minus_one_vec = self.pinned_vreg("minus_one_vec", VLEN)
        emit(("valu", ("vbroadcast", minus_one_vec, minus_one_const)))
        for vi in range(n_vectors):
            offset_imm = vi * VLEN
            offset_const = self.scratch_const(offset_imm)
            idx_base = self.new_vreg(f"idx_base_final_v{vi}")
            val_base = self.new_vreg(f"val_base_final_v{vi}")
            # Convert 1-based idx back to 0-based
            idx_0based = self.new_vreg_vec(f"idx_0based_v{vi}")
            emit(("valu", ("+", idx_0based, idx_vecs[vi], minus_one_vec)), vi=vi, rnd=rounds)
            emit(("flex_alu_add", ("+", idx_base, param_vregs["inp_indices_p"], offset_const, offset_imm)), vi=vi, rnd=rounds)
            emit(("flex_alu_add", ("+", val_base, param_vregs["inp_values_p"], offset_const, offset_imm)), vi=vi, rnd=rounds)
            emit(("store", ("vstore", idx_base, idx_0based)), vi=vi, rnd=rounds)
            emit(("store", ("vstore", val_base, val_vecs[vi])), vi=vi, rnd=rounds)

        # Prepend all setup loads to body for scheduling
        setup_slots = self.pending_const_loads + self.pending_mem_loads
        setup_tags = [{"vi": -1, "rnd": -1}] * len(setup_slots)
        self.pending_const_loads.clear()
        self.pending_mem_loads.clear()
        body = setup_slots + body
        body_tags = setup_tags + body_tags

        # Schedule + allocate in one pass
        allocator = ScratchAllocator(self.scratch_ptr, scratch_debug=self.scratch_debug)
        bundles, sched_meta = schedule(body, slot_limits or dict(SLOT_LIMITS), allocator, tags=body_tags)
        allocator.print_peak_info()
        physical_bundles, sched_meta = expand_valu_as_alu(bundles, sched_meta)

        # Extract per-(vi, rnd) timing and per-op stats from sched_meta
        self.vi_rnd_timing, self._op_stats = self._extract_timing(sched_meta)
        self._bundles_for_stats = physical_bundles

        body_instrs, slot_sched_info = self.build(physical_bundles, sched_meta)

        # Remap sched_info keys: bundle_idx -> actual PC (offset by existing instrs)
        pc_offset = len(self.instrs)
        self.slot_sched_info = {}
        for (bi, engine, si), meta in slot_sched_info.items():
            self.slot_sched_info[(pc_offset + bi, engine, si)] = meta

        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    slot_limits=None,
    plot: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds,
                     slot_limits=slot_limits)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)

    if plot:
        try:
            kb.print_timing_summary()
            kb.plot_timing()
        except ImportError as e:
            print(f"Plot import error: {e}")
        except Exception as e:
            print(f"Plot error: {e}")
            import traceback
            traceback.print_exc()

    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        sys.argv.remove("--plot")
        do_kernel_test(10, 16, 256, plot=True)
    else:
        unittest.main()
