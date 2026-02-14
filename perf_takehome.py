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
import copy
from collections import defaultdict
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
    if engine in ("debug", "barrier"):
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
            from collections import Counter
            prefixes = Counter()
            for v in vregs:
                name = v.name_hint
                prefix = name.rsplit('_v', 1)[0] if '_v' in name else name
                prefix = prefix.rsplit('_r', 1)[0] if '_r' in prefix else prefix
                prefixes[prefix] += 1
            for prefix, count in prefixes.most_common(10):
                print(f"      {prefix}: {count}")


def schedule_segment(slots, slot_limits, allocator, tags=None):
    """Schedule a segment of ops using list scheduling with integrated allocation.

    Builds a DAG from VReg def-use chains and greedily packs independent
    ops into cycles, respecting slot limits. Allocates physical addresses
    inline — no separate allocation pass needed.

    Args:
        slots: List of (engine, args) tuples with VRegs
        slot_limits: Dict of {engine: max_per_cycle}
        allocator: ScratchAllocator instance for physical address management
        tags: Optional list of dicts parallel to slots with metadata (vi, rnd, etc.)

    Returns:
        (bundles, sched_meta) where bundles have physical addresses
    """
    n = len(slots)
    if n == 0:
        return [], []

    # Step 1: Compute defs/uses per slot
    slot_defs = []
    slot_uses = []
    for engine, args in slots:
        d, u = get_defs_uses(engine, args)
        slot_defs.append(d)
        slot_uses.append(u)

    # Step 2: Build DAG
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

    def op_desc(i):
        """Human-readable description of op i using vreg names."""
        engine, args = slots[i]
        op_name = args[0] if args else ""
        # Get dest vreg name
        dest = ""
        if len(args) > 1 and isinstance(args[1], VReg):
            dest = args[1].name_hint or f"v{args[1].id}"
        return f"{engine} {op_name} {dest}".strip()

    def named_args(i):
        """Capture vreg names from original slot args (before address rewrite)."""
        engine, args = slots[i]
        named = []
        for a in args:
            if isinstance(a, VReg):
                named.append(a.name_hint or f"v{a.id}")
            else:
                named.append(a)
        return f"({engine} {' '.join(str(x) for x in named)})"

    # Step 2.5: Compute distance-to-nearest-load and distance-to-nearest-flow separately
    from collections import deque

    def bfs_dist(target_engines):
        dist = [n] * n
        queue = deque()
        for i in range(n):
            if slots[i][0] in target_engines:
                dist[i] = 0
                queue.append(i)
        while queue:
            j = queue.popleft()
            for pred in predecessors[j]:
                if dist[pred] > dist[j] + 1:
                    dist[pred] = dist[j] + 1
                    queue.append(pred)
        return dist

    dist_to_load = bfs_dist(("load",))
    dist_to_flow = bfs_dist(("flow",))
    dist_to_either = [min(dist_to_load[i], dist_to_flow[i]) for i in range(n)]

    # Step 2.6: Compute use counts for freeing
    use_count = defaultdict(int)
    for i in range(n):
        for vreg in slot_uses[i]:
            v = vreg.parent if (hasattr(vreg, 'parent') and vreg.parent is not None) else vreg
            if isinstance(v, VReg) and v.pinned_addr is None:
                use_count[v] += 1
    remaining_uses = dict(use_count)

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

    def consume_uses(i):
        """Decrement use counts, return vregs ready to free."""
        frees = []
        for v in slot_uses[i]:
            pv = resolve_parent(v)
            if isinstance(pv, VReg) and pv.pinned_addr is None:
                remaining_uses[pv] = remaining_uses.get(pv, 0) - 1
                if remaining_uses[pv] == 0:
                    frees.append(pv)
        return frees

    # Step 3: List scheduling with integrated allocation
    # Active distance map — updated each cycle based on what's starved
    active_dist = dist_to_either  # default

    def sched_key(i):
        return (active_dist[i], sort_key(i), i)

    ready = sorted([i for i in range(n) if in_degree[i] == 0], key=sched_key)
    bundles = []
    sched_meta = []
    partial_ops = {}
    stall_slot = 0
    stall_alloc = 0

    ready_cycle = {}
    op_scheduled_cycle = {}  # op index -> cycle it was scheduled
    op_to_id = {}  # op index -> unique op_id for flow events
    next_op_id = [0]  # mutable counter
    cycle_num = 0
    for i in ready:
        ready_cycle[i] = 0

    def assign_op_id(i):
        if i not in op_to_id:
            op_to_id[i] = next_op_id[0]
            next_op_id[0] += 1
        return op_to_id[i]

    def build_meta(i):
        """Build scheduling metadata dict for op i."""
        meta = {"ready": ready_cycle.get(i, 0), "sched": cycle_num, "deps": build_dep_info(i), "op_id": assign_op_id(i), "named": named_args(i), "pressure": sort_key(i), "dist_to_load": dist_to_load[i], "dist_to_flow": dist_to_flow[i]}
        if tags is not None and i < len(tags):
            meta.update(tags[i])
        return meta

    def build_dep_info(i):
        """Build dependency info for op i: list of {desc, sched_cycle, op_id} for each predecessor."""
        deps = []
        for pred in predecessors[i]:
            deps.append({
                "op": op_desc(pred),
                "cycle": op_scheduled_cycle.get(pred, -1),
                "op_id": assign_op_id(pred),
            })
        return deps

    while ready or partial_ops:
        bundle = []
        bundle_meta = []
        available = dict(slot_limits)
        scheduled_this_cycle = []
        remaining = []
        pending_frees = []

        # First: continue in-progress partial ops (priority)
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
                scheduled_this_cycle.append(i)
                op_scheduled_cycle[i] = cycle_num
                pending_frees.extend(consume_uses(i))

        # Adapt distance heuristic based on what's starved in the ready queue
        n_ready_loads = sum(1 for i in ready if slots[i][0] == "load")
        n_ready_flows = sum(1 for i in ready if slots[i][0] == "flow")
        load_starved = n_ready_loads < slot_limits.get("load", 2)
        flow_starved = n_ready_flows < slot_limits.get("flow", 1)
        if load_starved and not flow_starved:
            active_dist = dist_to_load
        elif flow_starved and not load_starved:
            active_dist = dist_to_flow
        else:
            active_dist = dist_to_either

        ready.sort(key=sched_key)

        # Schedule new ready ops
        for i in ready:
            engine = slots[i][0]

            # Check engine availability
            can_native = available.get(engine, 0) > 0
            can_promote = (engine == "valu"
                          and slots[i][1][0] not in ("vbroadcast", "multiply_add")
                          and available.get("alu", 0) > 0)

            if not can_native and not can_promote:
                remaining.append(i)
                stall_slot += 1
                continue

            # Try to allocate defs — if allocator says no room, defer
            if not try_alloc_defs(i):
                remaining.append(i)
                stall_alloc += 1
                if stall_alloc <= 5:
                    defs_info = [(v.name_hint, v.size) for v in slot_defs[i] if isinstance(v, VReg) and v.pinned_addr is None]
                    print(f"  alloc_fail cycle={cycle_num} op={op_desc(i)} usage={allocator.current_usage()} defs={defs_info}")
                continue

            # Commit
            meta = build_meta(i)
            if can_native:
                physical_args = allocator.rewrite_args(slots[i][1])
                bundle.append((engine, physical_args))
                bundle_meta.append(meta)
                available[engine] -= 1
                scheduled_this_cycle.append(i)
                op_scheduled_cycle[i] = cycle_num
                pending_frees.extend(consume_uses(i))
            else:
                # valu→alu promotion
                alu_avail = available.get("alu", 0)
                can_do = min(alu_avail, VLEN)
                physical_args = allocator.rewrite_args(slots[i][1])
                bundle.append(("valu_as_alu", physical_args, 0, can_do))
                bundle_meta.append(meta)
                available["alu"] -= can_do
                if can_do >= VLEN:
                    scheduled_this_cycle.append(i)
                    op_scheduled_cycle[i] = cycle_num
                    pending_frees.extend(consume_uses(i))
                else:
                    partial_ops[i] = can_do
                    op_scheduled_cycle[i] = cycle_num
                    # Don't consume uses yet — partial op still reading sources

        allocator.track_peak(cycle_num)  # snapshot BEFORE frees (true peak)

        # End of cycle: free dead vregs
        for vreg in pending_frees:
            allocator.free(vreg)

        # Newly ready ops
        newly_ready = []
        for i in scheduled_this_cycle:
            for succ in successors[i]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    newly_ready.append(succ)
                    ready_cycle[succ] = cycle_num + 1

        ready = sorted(remaining + newly_ready, key=sched_key)
        if bundle:
            bundles.append(bundle)
            sched_meta.append(bundle_meta)
            cycle_num += 1
        elif not partial_ops:
            if ready:
                raise RuntimeError(
                    f"Scheduler deadlock: {len(ready)} ready ops but none schedulable."
                )
            break

    print(f"Scheduler: {cycle_num} cycles, {n} ops, stalls: slot_full={stall_slot}, alloc_fail={stall_alloc}")
    return bundles, sched_meta


def schedule(slots, slot_limits=None, allocator=None, tags=None):
    """Schedule ops into bundles with integrated allocation.

    Splits on barrier pseudo-ops, schedules each segment independently.

    Args:
        slots: List of (engine, args) tuples, may include ("barrier", ())
        slot_limits: Optional dict overriding SLOT_LIMITS.
        allocator: ScratchAllocator instance.
        tags: Optional list of dicts parallel to slots with metadata (vi, rnd, etc.)

    Returns:
        (bundles, sched_meta) — bundles have physical addresses
    """
    if slot_limits is None:
        slot_limits = dict(SLOT_LIMITS)

    segments = []
    seg_tags_list = []
    current = []
    current_tags = []
    for idx, slot in enumerate(slots):
        if slot[0] == "barrier":
            segments.append(current)
            seg_tags_list.append(current_tags)
            current = []
            current_tags = []
        else:
            current.append(slot)
            current_tags.append(tags[idx] if tags is not None else None)
    segments.append(current)
    seg_tags_list.append(current_tags)

    bundles = []
    sched_meta = []
    for segment, seg_tags in zip(segments, seg_tags_list):
        seg_bundles, seg_meta = schedule_segment(segment, slot_limits, allocator, tags=seg_tags if tags is not None else None)
        bundles.extend(seg_bundles)
        sched_meta.extend(seg_meta)
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

    def pinned_const(self, val, name=None):
        """Get a pinned vreg for a constant value."""
        if val not in self.const_map:
            vreg_name = name or f"const_{val}"
            addr = self.alloc_scratch(vreg_name)
            self.add("load", ("const", addr, val))
            vreg = VReg(name_hint=vreg_name, size=1, pinned_addr=addr)
            self.const_map[val] = vreg
        return self.const_map[val]

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_in, hash_const_vecs):
        """Vectorized hash - operates on VLEN elements at once.
        Uses virtual registers. Returns (slots, val_out) where val_out is the result vreg."""
        slots = []
        val_vec = val_in

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec = hash_const_vecs[hi * 2]
            const3_vec = hash_const_vecs[hi * 2 + 1]

            # Create fresh vregs for this stage's outputs
            tmp1 = self.new_vreg_vec(f"hash{hi}_t1")
            tmp2 = self.new_vreg_vec(f"hash{hi}_t2")
            val_out = self.new_vreg_vec(f"hash{hi}_out")

            # tmp1 = op1(val, const1) and tmp2 = op3(val, const3) - independent
            slots.append(("valu", (op1, tmp1, val_vec, const1_vec)))
            slots.append(("valu", (op3, tmp2, val_vec, const3_vec)))
            slots.append(("valu", (op2, val_out, tmp1, tmp2)))

            val_vec = val_out  # Chain to next stage

        return slots, val_vec

    def preload_level(self, k, forest_values_p_addr):
        """Preload all node values at tree level k into broadcast vectors.

        Args:
            k: tree level (0 = root). Level k has 2^k nodes starting at
               tree index (2^k - 1) in the implicit binary tree.
            forest_values_p_addr: scalar scratch addr holding base pointer
               to tree values in main memory.

        Returns:
            (slots, broadcast_vregs) where:
            - slots: list of (engine, args) to append to body
            - broadcast_vregs: list of 2^k vector vregs, each containing
              one node value broadcast across VLEN elements

        Steps:
            1. Compute the memory base address for this level:
               level_base = forest_values_p + (2^k - 1)
               (2^k - 1 is a build-time constant, so use scratch_const)

            2. vload the 2^k values into scratch. Each vload brings in 8
               contiguous words, so you need ceil(2^k / 8) vloads. But here
               2^k might be < 8 (levels 0-2), so for those you'll need
               individual scalar loads instead.

            3. vbroadcast each loaded scalar value into its own vector vreg.
               These are the outputs the mux tree will consume.

        Watch out for:
            - Levels 0-2 have fewer than 8 nodes, so vload won't work cleanly.
              Use individual "load" ops for those, or load 8 and ignore extras.
            - Each vload needs a scalar vreg holding the memory address to
              load from. You'll need to increment that address between vloads.
        """
        slots = []
        n_nodes = 2 ** k

        level_offset = self.scratch_const(2**k - 1)
        level_base = self.new_vreg(f"level{k}_base")
        slots.append(("alu", ("+", level_base, forest_values_p_addr, level_offset)))
        vregs = []

        for i in range(max(n_nodes//8, 1)):
            current_offset = self.new_vreg(f"level{k}_offset_{i}")
            offset_constant = self.scratch_const(i*VLEN)
            slots.append(("alu", ("+", current_offset, level_base, offset_constant)))
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
        """Select each element's node value from preloaded level using a vselect mux tree.

        Args:
            broadcast_vregs: list of 2^k vector vregs from preload_level,
                each containing one node value broadcast across VLEN.
            idx_vreg: vector vreg holding current tree indices for this
                vector group (each element is a tree node index).
            k: tree level (broadcast_vregs has 2^k entries).

        Returns:
            (slots, result_vreg) where result_vreg is a vector vreg
            containing the selected node value per element.

        Steps:
            1. Compute position within level:
               position = idx - (2^k - 1)
               This is a vector subtract using a broadcast constant.

            2. Mux tree, k stages from MSB to LSB:
               For stage s (0 to k-1):
                 a. Extract bit (k-1-s) from position:
                    bit = (position >> (k-1-s)) & 1
                    This needs a shift and an AND, both valu ops.

                 b. vselect pairs of candidates:
                    For each pair (candidates[2j], candidates[2j+1]):
                      result = vselect(bit, candidates[2j+1], candidates[2j])
                    This halves the candidate list each stage.

               Note: vselect(cond, a, b) returns a[i] if cond[i]!=0, else b[i].
               So bit=1 picks candidates[2j+1] (right child path),
               bit=0 picks candidates[2j] (left child path).

            3. After k stages, one candidate remains — that's the result.

        Watch out for:
            - k=0 is a special case: only 1 broadcast vreg, no mux needed.
              Just return it directly.
            - Shift amounts are compile-time constants, so use scratch_const.
        """
        def shift_bit(idx_vreg, num_bits, k, i):
            # returns (slots, bit_vreg) where bit_vreg has the extracted bit per element
            s = []
            shift_const_vec = self.pinned_vreg(f"shift_{num_bits}_vec", VLEN)
            s.append(("valu", ("vbroadcast", shift_const_vec, self.scratch_const(num_bits))))
            shifted = self.new_vreg_vec(f"shifted_{num_bits}_stage{k}_vec{i}")
            s.append(("valu", (">>", shifted, idx_vreg, shift_const_vec)))
            bit = self.new_vreg_vec(f"bit_{num_bits}_stage{k}_vec{i}")
            s.append(("valu", ("&", bit, shifted, self.pinned_vreg("one_vec", VLEN))))
            return s, bit
        
        slots = []
        remaining_vregs = list(broadcast_vregs)
        if k == 0:
            return slots, broadcast_vregs[0]

        if k == 1:
            # Special case: just need (idx - 1) & 1, no shift needed
            adjusted_idx = self.new_vreg_vec(f"mux1_adjusted_idx_vec{index}")
            slots.append(("valu", ("-", adjusted_idx, idx_vreg, self.pinned_vreg("level_start1_vec", VLEN))))
            bit = self.new_vreg_vec(f"bit_0_stage0_vec{index}")
            slots.append(("valu", ("&", bit, adjusted_idx, self.pinned_vreg("one_vec", VLEN))))
            result = self.new_vreg_vec(f"mux_stage0_vec{index}_0")
            slots.append(("flow", ("vselect", result, bit, broadcast_vregs[1], broadcast_vregs[0])))
            return slots, result

        adjusted_idx = self.new_vreg_vec(f"mux{k}_adjusted_idx_vec{index}")
        slots.append(("valu", ("-", adjusted_idx, idx_vreg, self.pinned_vreg(f"level_start{k}_vec", VLEN))))

        for stage in range(k):
            shift_slots, condition_vreg = shift_bit(adjusted_idx, stage, stage, index)
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
        """
        Allocate scratch space and set up initial constants.
        Collects loads into pending lists for batch emission.
        Returns (one_const, two_const).
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
        self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        for _, val1, _, _, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)

        return one_const, two_const, param_vregs

    def emit_pending_loads(self):
        """Batch-emit all pending const and mem loads, packed 2 per cycle."""
        # Const loads first (all independent)
        for i in range(0, len(self.pending_const_loads), 2):
            chunk = self.pending_const_loads[i:i+2]
            self.instrs.append({"load": chunk})

        # Mem loads second (each depends on its const, but independent of each other)
        for i in range(0, len(self.pending_mem_loads), 2):
            chunk = self.pending_mem_loads[i:i+2]
            self.instrs.append({"load": chunk})

        self.pending_const_loads.clear()
        self.pending_mem_loads.clear()

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
        slot_limits=None,
    ):
        """
        Vectorized kernel using virtual registers.
        Each write creates a fresh vreg (SSA form).
        """
        # Setup phase: allocate scratch and register constants
        one_const, two_const, param_vregs = self.setup_kernel_scratch_and_constants()

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
        one_vec = self.pinned_vreg("one_vec", VLEN)
        two_vec = self.pinned_vreg("two_vec", VLEN)
        n_nodes_vec = self.pinned_vreg("n_nodes_vec", VLEN)
        forest_p_vec = self.pinned_vreg("forest_p_vec", VLEN)

        emit(("valu", ("vbroadcast", one_vec, one_const)))
        emit(("valu", ("vbroadcast", two_vec, two_const)))
        emit(("valu", ("vbroadcast", n_nodes_vec, param_vregs["n_nodes"])))
        emit(("valu", ("vbroadcast", forest_p_vec, param_vregs["forest_values_p"])))
        for k in range(3):
            level_start_vec = self.pinned_vreg(f"level_start{k}_vec", VLEN)
            emit(("valu", ("vbroadcast", level_start_vec, self.scratch_const(2**k - 1))))

        # Pre-broadcast all 12 hash constants (pinned since used every iteration)
        hash_const_vecs = []
        for hi, (_, val1, _, _, val3) in enumerate(HASH_STAGES):
            const1_vec = self.pinned_vreg(f"hash_c1_{hi}_vec", VLEN)
            const3_vec = self.pinned_vreg(f"hash_c3_{hi}_vec", VLEN)
            emit(("valu", ("vbroadcast", const1_vec, self.scratch_const(val1))))
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
            offset_const = self.scratch_const(vi * VLEN)
            idx_base = self.new_vreg(f"idx_base_init_v{vi}")
            val_base = self.new_vreg(f"val_base_init_v{vi}")
            emit(("alu", ("+", idx_base, param_vregs["inp_indices_p"], offset_const)), vi=vi, rnd=-1)
            emit(("alu", ("+", val_base, param_vregs["inp_values_p"], offset_const)), vi=vi, rnd=-1)

            idx_v = self.new_vreg_vec(f"idx_init_v{vi}")
            val_v = self.new_vreg_vec(f"val_init_v{vi}")
            emit(("load", ("vload", idx_v, idx_base)), vi=vi, rnd=-1)
            emit(("load", ("vload", val_v, val_base)), vi=vi, rnd=-1)
            idx_vecs.append(idx_v)
            val_vecs.append(val_v)

        # Main loop: all rounds, reading/writing scratch VRegs (no memory round-trip)
        cached_broadcasts = {}
        for rnd in range(rounds):
            new_idx_vecs = []
            new_val_vecs = []
            k = rnd % (forest_height + 1)
            if k <= 1:
                if k in cached_broadcasts:
                    broadcast_vregs = cached_broadcasts[k]
                else:
                    preload_slots, broadcast_vregs = self.preload_level(k, param_vregs["forest_values_p"])
                    emit_all(preload_slots, rnd=rnd)
                    cached_broadcasts[k] = broadcast_vregs

            # Optimal mux/gather split: balance flow (mux) vs load (gather)
            # m = 4n / (2^k + 3), rounded to nearest int
            mux_count = {0: n_vectors, 1: n_vectors - 2}

            for vi in range(n_vectors):
                idx_loaded = idx_vecs[vi]
                val_loaded = val_vecs[vi]

                # Compute gather addresses: addr = forest_p + idx
                if k <= 1 and vi < mux_count[k]:
                    select_slots, node_val = self.build_mux_select(broadcast_vregs, idx_loaded, k, vi)
                    emit_all(select_slots, vi=vi, rnd=rnd)

                else:
                    addr_vec = self.new_vreg_vec(f"addr_r{rnd}_v{vi}")
                    emit(("valu", ("+", addr_vec, forest_p_vec, idx_loaded)), vi=vi, rnd=rnd)

                    # Gather node values from tree (still from main memory)
                    gather_slots, node_val = self.build_gather(addr_vec, f"node_r{rnd}_v{vi}")
                    emit_all(gather_slots, vi=vi, rnd=rnd)

                # val = val ^ node_val
                val_xored = self.new_vreg_vec(f"xor_r{rnd}_v{vi}")
                emit(("valu", ("^", val_xored, val_loaded, node_val)), vi=vi, rnd=rnd)

                # val = myhash(val)
                hash_slots, val_hashed = self.build_vhash(val_xored, hash_const_vecs)
                emit_all(hash_slots, vi=vi, rnd=rnd)

                # idx = 2*idx + 1 + (val & 1)
                parity = self.new_vreg_vec(f"parity_r{rnd}_v{vi}")
                idx_doubled_plus1 = self.new_vreg_vec(f"idx2p1_r{rnd}_v{vi}")
                idx_next = self.new_vreg_vec(f"idx_next_r{rnd}_v{vi}")

                emit(("valu", ("&", parity, val_hashed, one_vec)), vi=vi, rnd=rnd)
                emit(("valu", ("multiply_add", idx_doubled_plus1, idx_loaded, two_vec, one_vec)), vi=vi, rnd=rnd)
                emit(("valu", ("+", idx_next, idx_doubled_plus1, parity)), vi=vi, rnd=rnd)

                # idx = idx * (idx < n_nodes) -- wraps to 0 if out of bounds
                in_bounds = self.new_vreg_vec(f"inbounds_r{rnd}_v{vi}")
                idx_wrapped = self.new_vreg_vec(f"idx_wrap_r{rnd}_v{vi}")
                emit(("valu", ("<", in_bounds, idx_next, n_nodes_vec)), vi=vi, rnd=rnd)
                emit(("valu", ("*", idx_wrapped, idx_next, in_bounds)), vi=vi, rnd=rnd)

                new_idx_vecs.append(idx_wrapped)
                new_val_vecs.append(val_hashed)

            idx_vecs = new_idx_vecs
            val_vecs = new_val_vecs

        # Store final results back to memory (once)
        for vi in range(n_vectors):
            offset_const = self.scratch_const(vi * VLEN)
            idx_base = self.new_vreg(f"idx_base_final_v{vi}")
            val_base = self.new_vreg(f"val_base_final_v{vi}")
            emit(("alu", ("+", idx_base, param_vregs["inp_indices_p"], offset_const)), vi=vi, rnd=rounds)
            emit(("alu", ("+", val_base, param_vregs["inp_values_p"], offset_const)), vi=vi, rnd=rounds)
            emit(("store", ("vstore", idx_base, idx_vecs[vi])), vi=vi, rnd=rounds)
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
        bundles, sched_meta = schedule(body, slot_limits, allocator, tags=body_tags)
        allocator.print_peak_info()
        physical_bundles, sched_meta = expand_valu_as_alu(bundles, sched_meta)

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
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds,
                     slot_limits=slot_limits)
    # print(kb.instrs)

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
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

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
    unittest.main()
