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


def schedule_segment(slots, slot_limits, scratch_budget=SCRATCH_SIZE):
    """Schedule a segment of ops (no barriers) using list scheduling.

    Builds a DAG from VReg def-use chains and greedily packs independent
    ops into cycles respecting slot limits. Uses pressure-aware scheduling
    to avoid exceeding scratch space.

    Args:
        slots: List of (engine, args) tuples with VRegs
        slot_limits: Dict of {engine: max_per_cycle}
        scratch_budget: Max scratch words available for dynamic vreg allocation

    Returns:
        List of bundles (list of list of slots)
    """
    n = len(slots)
    if n == 0:
        return []

    # Step 1: Compute defs/uses per slot
    slot_defs = []
    slot_uses = []
    for engine, args in slots:
        d, u = get_defs_uses(engine, args)
        slot_defs.append(d)
        slot_uses.append(u)

    # Step 2: Build DAG
    # Track which slots define each VReg (handles load_offset partial defs)
    vreg_definers = defaultdict(list)  # vreg -> [slot indices that define it]
    successors = [[] for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        # Add edges: i depends on all definers of VRegs it uses
        preds_for_i = set()
        for vreg in slot_uses[i]:
            # Check both the vreg itself and its parent (for sub_vregs)
            lookup_vregs = [vreg]
            if hasattr(vreg, 'parent') and vreg.parent is not None:
                lookup_vregs.append(vreg.parent)
            for lv in lookup_vregs:
                for definer in vreg_definers[lv]:
                    if definer not in preds_for_i:
                        preds_for_i.add(definer)
                        successors[definer].append(i)
                        in_degree[i] += 1

        # Register this slot's defs
        for vreg in slot_defs[i]:
            vreg_definers[vreg].append(i)

    # Step 2.5: Compute use counts and pressure tracking for scratch-aware scheduling
    use_count = defaultdict(int)
    for i in range(n):
        for vreg in slot_uses[i]:
            # Track the physical vreg (parent if sub_vreg)
            v = vreg.parent if (hasattr(vreg, 'parent') and vreg.parent is not None) else vreg
            if isinstance(v, VReg) and v.pinned_addr is None:
                use_count[v] += 1

    remaining_uses = dict(use_count)
    live_vregs = set()
    estimated_pressure = 0

    def pressure_delta(i):
        """Compute scratch pressure change if slot i is scheduled."""
        added = 0
        for v in slot_defs[i]:
            if isinstance(v, VReg) and v.pinned_addr is None and v not in live_vregs:
                added += v.size
        freed = 0
        for v in slot_uses[i]:
            pv = v.parent if (hasattr(v, 'parent') and v.parent is not None) else v
            if isinstance(pv, VReg) and pv.pinned_addr is None and pv in live_vregs:
                if remaining_uses.get(pv, 0) == 1:
                    freed += pv.size
        return added - freed

    pending_frees = []  # vregs to free at end of cycle (matches allocator behavior)

    def update_pressure_defs(i):
        """Track new vreg definitions (pressure increases). Called per-op."""
        nonlocal estimated_pressure
        for v in slot_defs[i]:
            if isinstance(v, VReg) and v.pinned_addr is None and v not in live_vregs:
                live_vregs.add(v)
                estimated_pressure += v.size

    def update_pressure_uses(i):
        """Track vreg uses and queue frees. Called per-op."""
        for v in slot_uses[i]:
            pv = v.parent if (hasattr(v, 'parent') and v.parent is not None) else v
            if isinstance(pv, VReg) and pv.pinned_addr is None:
                remaining_uses[pv] = remaining_uses.get(pv, 0) - 1
                if remaining_uses[pv] == 0 and pv in live_vregs:
                    pending_frees.append(pv)

    def flush_frees():
        """Apply pending frees at end of cycle (matches allocator's free_dead_vregs)."""
        nonlocal estimated_pressure
        for pv in pending_frees:
            if pv in live_vregs:
                live_vregs.remove(pv)
                estimated_pressure -= pv.size
        pending_frees.clear()

    # Step 3: List scheduling with partial valu→alu promotion
    ready = sorted([i for i in range(n) if in_degree[i] == 0])
    bundles = []
    partial_ops = {}  # slot_index -> elements_done_so_far

    while ready or partial_ops:
        bundle = []
        available = dict(slot_limits)
        scheduled_this_cycle = []
        remaining = []

        # First: continue in-progress partial ops (priority)
        for i in list(partial_ops):
            alu_avail = available.get("alu", 0)
            if alu_avail <= 0:
                break
            done_so_far = partial_ops[i]
            can_do = min(alu_avail, VLEN - done_so_far)
            bundle.append(("valu_as_alu", slots[i][1], done_so_far, can_do))
            available["alu"] -= can_do
            partial_ops[i] = done_so_far + can_do
            if partial_ops[i] >= VLEN:
                del partial_ops[i]
                scheduled_this_cycle.append(i)
                update_pressure_defs(i)
                update_pressure_uses(i)

        # Sort ready ops: prefer those that reduce pressure (negative delta first)
        ready.sort(key=pressure_delta)

        # Then: schedule new ready ops
        for i in ready:
            engine = slots[i][0]

            # Backpressure: if this op would define new vregs pushing over budget,
            # delay it until other ops free space
            added = sum(v.size for v in slot_defs[i]
                        if isinstance(v, VReg) and v.pinned_addr is None and v not in live_vregs)
            if added > 0 and estimated_pressure + added > scratch_budget:
                remaining.append(i)
                continue

            if available.get(engine, 0) > 0:
                bundle.append(slots[i])
                available[engine] -= 1
                scheduled_this_cycle.append(i)
                update_pressure_defs(i)
                update_pressure_uses(i)
            elif (engine == "valu"
                  and slots[i][1][0] not in ("vbroadcast", "multiply_add")
                  and available.get("alu", 0) > 0):
                # Partial or full valu→alu promotion
                alu_avail = available.get("alu", 0)
                can_do = min(alu_avail, VLEN)
                bundle.append(("valu_as_alu", slots[i][1], 0, can_do))
                available["alu"] -= can_do
                if can_do >= VLEN:
                    scheduled_this_cycle.append(i)
                    update_pressure_defs(i)
                    update_pressure_uses(i)
                else:
                    partial_ops[i] = can_do
                    update_pressure_defs(i)  # Allocator allocates on first partial cycle
            else:
                remaining.append(i)

        # Flush frees at end of cycle (matches allocator's free_dead_vregs behavior)
        flush_frees()


        # Newly ready ops (for next cycle)
        newly_ready = []
        for i in scheduled_this_cycle:
            for succ in successors[i]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    newly_ready.append(succ)

        # Next cycle's ready list: unscheduled from this cycle + newly unblocked
        ready = remaining + sorted(newly_ready)
        if bundle:
            bundles.append(bundle)
        elif not partial_ops:
            if ready:
                raise RuntimeError(
                    f"Scheduler deadlock: {len(ready)} ready ops but none schedulable. "
                    f"estimated_pressure={estimated_pressure}/{scratch_budget}, "
                    f"ready deltas: {[(i, pressure_delta(i)) for i in ready[:5]]}"
                )
            break  # all ops scheduled

    return bundles


def schedule(slots, slot_limits=None, scratch_budget=SCRATCH_SIZE):
    """Schedule ops into bundles with configurable slot limits.

    Splits on barrier pseudo-ops, schedules each segment independently.

    Args:
        slots: List of (engine, args) tuples, may include ("barrier", ())
        slot_limits: Optional dict overriding SLOT_LIMITS. Keys are engine
                     names, values are max ops per cycle for that engine.
        scratch_budget: Max scratch words available for dynamic vreg allocation.

    Returns:
        List of bundles (list of list of slots)
    """
    if slot_limits is None:
        slot_limits = dict(SLOT_LIMITS)

    # Split on barriers
    segments = []
    current = []
    for slot in slots:
        if slot[0] == "barrier":
            segments.append(current)
            current = []
        else:
            current.append(slot)
    segments.append(current)

    # Schedule each segment independently
    bundles = []
    for segment in segments:
        bundles.extend(schedule_segment(segment, slot_limits, scratch_budget))
    return bundles


def expand_valu_as_alu(bundles):
    """Expand valu_as_alu slots into scalar alu ops.

    Each valu_as_alu slot is a 4-tuple: (engine, args, start, count)
    specifying which element range to expand.
    Must be called AFTER register allocation (physical addresses assigned).
    """
    result = []
    for bundle in bundles:
        new_bundle = []
        for slot in bundle:
            if slot[0] == "valu_as_alu":
                args, start, count = slot[1], slot[2], slot[3]
                op = args[0]
                dest = args[1]
                sources = args[2:]
                for i in range(start, start + count):
                    new_args = (op, dest + i) + tuple(s + i for s in sources)
                    new_bundle.append(("alu", new_args))
            else:
                new_bundle.append((slot[0], slot[1]))
        result.append(new_bundle)
    return result


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
        return DebugInfo(scratch_map=self.scratch_debug)

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

    def allocate_vregs(self, bundles):
        """
        Allocate physical scratch addresses for virtual registers.
        Uses liveness analysis to reuse scratch space.

        Args:
            bundles: List of bundles, where each bundle is a list of slots.
                     Each bundle represents one cycle.

        Returns:
            List of bundles with physical addresses instead of VRegs.
        """
        # Step 1: Compute liveness - find last cycle where each vreg is used
        last_use = {}  # vreg -> cycle index of last use

        def collect_vregs(args):
            """Collect all VRegs from instruction arguments."""
            for a in args:
                if isinstance(a, VReg) and a.pinned_addr is None:
                    yield a

        for cycle_idx, bundle in enumerate(bundles):
            for slot in bundle:
                engine, args = slot[0], slot[1]
                if engine == "debug":
                    continue
                for vreg in collect_vregs(args):
                    # Sub-vregs extend their parent's liveness
                    if vreg.parent is not None:
                        last_use[vreg.parent] = max(last_use.get(vreg.parent, 0), cycle_idx)
                    else:
                        last_use[vreg] = cycle_idx

        # Step 2: Allocate with scanning — no watermarks
        # Vectors (size 8): scan left-to-right for free aligned blocks
        # Scalars (size 1): scan right-to-left for free slots
        # Pressure = actual occupied words, no watermark drift
        vreg_to_addr = {}
        occupied = [False] * SCRATCH_SIZE
        # Mark pre-allocated region as occupied
        for i in range(self.scratch_ptr):
            occupied[i] = True

        def alloc_vector():
            for a in range(self.scratch_ptr, SCRATCH_SIZE - VLEN + 1):
                if not any(occupied[a:a + VLEN]):
                    for i in range(VLEN):
                        occupied[a + i] = True
                    return a
            assert False, f"Out of scratch: no free {VLEN}-word block"

        def alloc_scalar():
            for a in range(SCRATCH_SIZE - 1, self.scratch_ptr - 1, -1):
                if not occupied[a]:
                    occupied[a] = True
                    return a
            assert False, "Out of scratch: no free scalar slot"

        def free_addr(addr, size):
            for i in range(size):
                occupied[addr + i] = False

        def get_addr(vreg):
            if isinstance(vreg, VReg):
                if vreg.pinned_addr is not None:
                    return vreg.pinned_addr
                if vreg.parent is not None:
                    return get_addr(vreg.parent) + vreg.offset
                if vreg not in vreg_to_addr:
                    if vreg.size == 1:
                        addr = alloc_scalar()
                    else:
                        addr = alloc_vector()
                    vreg_to_addr[vreg] = addr
                return vreg_to_addr[vreg]
            return vreg

        def free_dead_vregs(cycle_idx):
            """Free addresses of vregs whose last use was this cycle."""
            dead = [v for v, last in last_use.items() if last == cycle_idx and v in vreg_to_addr]
            for vreg in dead:
                free_addr(vreg_to_addr[vreg], vreg.size)

        def rewrite_slot(slot):
            engine, args = slot[0], slot[1]
            if engine == "debug":
                return slot
            new_args = tuple(get_addr(a) if isinstance(a, (VReg, int)) else a for a in args)
            return (engine, new_args) + slot[2:]  # preserve start/count for valu_as_alu

        physical_bundles = []
        peak_usage = 0
        peak_cycle = 0
        for cycle_idx, bundle in enumerate(bundles):
            self._debug_alloc_state = (vreg_to_addr, last_use, cycle_idx)
            physical_bundle = [rewrite_slot(slot) for slot in bundle]
            physical_bundles.append(physical_bundle)
            free_dead_vregs(cycle_idx)

            # Track net live scratch usage (just count occupied words)
            net_usage = sum(occupied[self.scratch_ptr:])
            if net_usage > peak_usage:
                peak_usage = net_usage
                peak_cycle = cycle_idx
                peak_engines = [slot[0] for slot in bundle]
                # Snapshot live vregs at peak
                freed_vregs = set()
                for v, last in last_use.items():
                    if last <= cycle_idx and v in vreg_to_addr:
                        freed_vregs.add(id(v))
                live_vregs = [(v, vreg_to_addr[v]) for v in vreg_to_addr if id(v) not in freed_vregs]
                peak_live = live_vregs

        print(f"Scratch: peak={peak_usage}/{SCRATCH_SIZE - self.scratch_ptr} at cycle {peak_cycle} (ops: {peak_engines})")
        print(f"  Live vregs at peak ({len(peak_live)}):")
        by_size = {}
        for v, addr in peak_live:
            by_size.setdefault(v.size, []).append(v)
        for size, vregs in sorted(by_size.items()):
            words = len(vregs) * size
            print(f"    size={size}: {len(vregs)} vregs ({words} words)")
            # Show names grouped by prefix
            from collections import Counter
            prefixes = Counter()
            for v in vregs:
                name = v.name_hint
                # Extract prefix (everything before the last _v or _r number)
                prefix = name.rsplit('_v', 1)[0] if '_v' in name else name
                prefix = prefix.rsplit('_r', 1)[0] if '_r' in prefix else prefix
                prefixes[prefix] += 1
            for prefix, count in prefixes.most_common(10):
                print(f"      {prefix}: {count}")

        return physical_bundles

    def build(self, bundles: list[list[tuple[Engine, tuple]]]):
        """
        Convert bundles of slots into instruction format.

        Args:
            bundles: List of bundles, where each bundle is a list of (engine, args) slots.

        Returns:
            List of instruction dicts: [{engine: [slot, ...], ...}, ...]
        """
        instrs = []
        for bundle in bundles:
            instr = defaultdict(list)
            for engine, args in bundle:
                instr[engine].append(args)
            instrs.append(dict(instr))
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        if self.scratch_ptr > SCRATCH_SIZE:
            print(f"OUT OF SCRATCH: ptr={self.scratch_ptr}/{SCRATCH_SIZE}, "
                  f"allocating '{name}' (size={length})")
            # Dump live vregs from allocator context
            dbg = getattr(self, '_debug_alloc_state', None)
            if dbg:
                vreg_to_addr, last_use, cycle_idx = dbg
                from collections import Counter
                live = [(v, vreg_to_addr[v]) for v in vreg_to_addr if last_use.get(v, 0) >= cycle_idx]
                by_size = {}
                for v, addr in live:
                    by_size.setdefault(v.size, []).append(v)
                print(f"  Live vregs at cycle {cycle_idx} ({len(live)}):")
                for size, vregs in sorted(by_size.items()):
                    words = len(vregs) * size
                    print(f"    size={size}: {len(vregs)} vregs ({words} words)")
                    prefixes = Counter()
                    for v in vregs:
                        n = v.name_hint
                        prefix = n.rsplit('_v', 1)[0] if '_v' in n else n
                        prefix = prefix.rsplit('_r', 1)[0] if '_r' in prefix else prefix
                        prefixes[prefix] += 1
                    for prefix, count in prefixes.most_common(15):
                        print(f"      {prefix}: {count}")
                        if count > 5:
                            samples = [v for v in vregs if prefix in v.name_hint][:5]
                            for v in samples:
                                print(f"        {v} last_use={last_use.get(v, '???')} (current cycle={cycle_idx})")
            assert False, "Out of scratch space"
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

        # Pinned vregs for broadcast constants (allocated once, used throughout)
        one_vec = self.pinned_vreg("one_vec", VLEN)
        two_vec = self.pinned_vreg("two_vec", VLEN)
        n_nodes_vec = self.pinned_vreg("n_nodes_vec", VLEN)
        forest_p_vec = self.pinned_vreg("forest_p_vec", VLEN)

        body.append(("valu", ("vbroadcast", one_vec, one_const)))
        body.append(("valu", ("vbroadcast", two_vec, two_const)))
        body.append(("valu", ("vbroadcast", n_nodes_vec, param_vregs["n_nodes"])))
        body.append(("valu", ("vbroadcast", forest_p_vec, param_vregs["forest_values_p"])))
        for k in range(7):
            level_start_vec = self.pinned_vreg(f"level_start{k}_vec", VLEN)
            body.append(("valu", ("vbroadcast", level_start_vec, self.scratch_const(2**k - 1))))

        # Pre-broadcast all 12 hash constants (pinned since used every iteration)
        hash_const_vecs = []
        for hi, (_, val1, _, _, val3) in enumerate(HASH_STAGES):
            const1_vec = self.pinned_vreg(f"hash_c1_{hi}_vec", VLEN)
            const3_vec = self.pinned_vreg(f"hash_c3_{hi}_vec", VLEN)
            body.append(("valu", ("vbroadcast", const1_vec, self.scratch_const(val1))))
            body.append(("valu", ("vbroadcast", const3_vec, self.scratch_const(val3))))
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
            body.append(("alu", ("+", idx_base, param_vregs["inp_indices_p"], offset_const)))
            body.append(("alu", ("+", val_base, param_vregs["inp_values_p"], offset_const)))

            idx_v = self.new_vreg_vec(f"idx_init_v{vi}")
            val_v = self.new_vreg_vec(f"val_init_v{vi}")
            body.append(("load", ("vload", idx_v, idx_base)))
            body.append(("load", ("vload", val_v, val_base)))
            idx_vecs.append(idx_v)
            val_vecs.append(val_v)

        # Main loop: all rounds, reading/writing scratch VRegs (no memory round-trip)
        for rnd in range(rounds):
            new_idx_vecs = []
            new_val_vecs = []
            k = rnd % (forest_height + 1)
            if k <= 4:
                preload_slots, broadcast_vregs = self.preload_level(k, param_vregs["forest_values_p"])
                # if rnd < 10:
                body.extend(preload_slots)

            # Optimal mux/gather split: balance flow (mux) vs load (gather)
            # m = 4n / (2^k + 3), rounded to nearest int
            mux_count = {0: n_vectors, 1: round(4 * n_vectors / 5) - 4, 2: round(4 * n_vectors / 7) - 5, 3: round(4 * n_vectors / 9) - 4, 4: 2}

            for vi in range(n_vectors):
                idx_loaded = idx_vecs[vi]
                val_loaded = val_vecs[vi]

                # Compute gather addresses: addr = forest_p + idx
                if k <= 4 and vi < mux_count[k]:
                    select_slots, node_val = self.build_mux_select(broadcast_vregs, idx_loaded, k, vi)
                    body.extend(select_slots)

                else:
                    addr_vec = self.new_vreg_vec(f"addr_r{rnd}_v{vi}")
                    body.append(("valu", ("+", addr_vec, forest_p_vec, idx_loaded)))

                    # Gather node values from tree (still from main memory)
                    gather_slots, node_val = self.build_gather(addr_vec, f"node_r{rnd}_v{vi}")
                    body.extend(gather_slots)

                # val = val ^ node_val
                val_xored = self.new_vreg_vec(f"xor_r{rnd}_v{vi}")
                body.append(("valu", ("^", val_xored, val_loaded, node_val)))

                # val = myhash(val)
                hash_slots, val_hashed = self.build_vhash(val_xored, hash_const_vecs)
                body.extend(hash_slots)

                # idx = 2*idx + 1 + (val & 1)
                parity = self.new_vreg_vec(f"parity_r{rnd}_v{vi}")
                idx_doubled_plus1 = self.new_vreg_vec(f"idx2p1_r{rnd}_v{vi}")
                idx_next = self.new_vreg_vec(f"idx_next_r{rnd}_v{vi}")

                body.append(("valu", ("&", parity, val_hashed, one_vec)))
                body.append(("valu", ("multiply_add", idx_doubled_plus1, idx_loaded, two_vec, one_vec)))
                body.append(("valu", ("+", idx_next, idx_doubled_plus1, parity)))

                # idx = idx * (idx < n_nodes) -- wraps to 0 if out of bounds
                in_bounds = self.new_vreg_vec(f"inbounds_r{rnd}_v{vi}")
                idx_wrapped = self.new_vreg_vec(f"idx_wrap_r{rnd}_v{vi}")
                body.append(("valu", ("<", in_bounds, idx_next, n_nodes_vec)))
                body.append(("valu", ("*", idx_wrapped, idx_next, in_bounds)))

                new_idx_vecs.append(idx_wrapped)
                new_val_vecs.append(val_hashed)

            idx_vecs = new_idx_vecs
            val_vecs = new_val_vecs

        # Store final results back to memory (once)
        for vi in range(n_vectors):
            offset_const = self.scratch_const(vi * VLEN)
            idx_base = self.new_vreg(f"idx_base_final_v{vi}")
            val_base = self.new_vreg(f"val_base_final_v{vi}")
            body.append(("alu", ("+", idx_base, param_vregs["inp_indices_p"], offset_const)))
            body.append(("alu", ("+", val_base, param_vregs["inp_values_p"], offset_const)))
            body.append(("store", ("vstore", idx_base, idx_vecs[vi])))
            body.append(("store", ("vstore", val_base, val_vecs[vi])))

        # Prepend all setup loads to body for scheduling
        setup_slots = self.pending_const_loads + self.pending_mem_loads
        self.pending_const_loads.clear()
        self.pending_mem_loads.clear()
        body = setup_slots + body

        # Schedule: pack independent ops into same cycle
        # Budget = total scratch minus what's already allocated for constants/params
        scratch_budget = SCRATCH_SIZE - self.scratch_ptr
        bundles = schedule(body, slot_limits, scratch_budget)

        # Allocate physical addresses for virtual registers
        physical_bundles = self.allocate_vregs(bundles)
        physical_bundles = expand_valu_as_alu(physical_bundles)

        body_instrs = self.build(physical_bundles)
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
