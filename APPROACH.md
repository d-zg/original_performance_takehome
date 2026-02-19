# Approach

**Result**: 1,118 cycles (131.5x speedup over 147,734 baseline), 9/9 submission tests passing.

## Infrastructure

After the easy win of vectorizing the kernel, gathering node values for each vector, I initially tried packing the flat op list into VLIW bundles by hand. This hit a wall around ~2,500-3000 cycles -- manually tracking which ops could go together, managing scratch addresses, and avoiding conflicts was tedious and I couldn't pack them well. To handle this, we built the following infrastructure:  

1. **Virtual registers (SSA-style)**: Each value gets a fresh vreg ID on definition. An allocator assigns physical scratch addresses based on liveness, reusing space as vregs die. This decoupled code generation from scratch management and let the scheduler build a proper dependency DAG from def-use chains.

2. **List scheduler with integrated allocation**: A greedy scheduler packs independent ops into VLIW bundles respecting slot limits (6 valu, 12 alu, 2 load, 2 store, 1 flow per cycle). Allocation happens inline during scheduling -- if an op would overflow scratch, it gets deferred rather than crashing.


## Key Optimizations

### 1. Reducing operation count

Most of these were discovered by doing detailed operation-by-operation walkthroughs of the hot loop and repeatedly asking Claude "why are we doing this?" about each step.

- **multiply_add fusion**: `(a + const) + (a << N)` collapses to a single `multiply_add(a, 2^N+1, const)`. Applied to the hash function (6 stages, 3 ops each -> 1 op for fusable stages) and index updates.
- **1-based tree indexing**: The index update `idx = 2*idx + 1 + (val & 1)` becomes `idx' = 2*idx' + parity` saving one VALU op per vector per round by making the +1 implicit. We pay a minor setup cost (computing `forest_values_p - 1` and broadcasting it for gather addressing) but save one op per vector per non-leaf round.
- **Eliminating bounds checks**: Since forest_height is known at compile time, we know exactly which round a walker reaches the leaves. No runtime comparison needed.
- **No recomputation on wrap-around**: At leaf rounds, indices reset to 1 (the root in 1-based indexing) with zero ops, just reuse the constant vector.
- **Bit extraction without shift**: For mux tree position bits, broadcast vectors of {1, 2, 4, 8} and AND directly instead of shifting then masking.
- **Eliminating memory round-trips**: The baseline loads indices and values from memory every round and stores them back. Instead, keep them in scratch vregs across all 16 rounds — load once at the start, store once at the end. This required rewriting the ops to chain vregs between rounds rather than going through memory. Same idea for hash constants, broadcast vectors, and mux tree node values — anything shared across rounds gets computed once and reused.
- **Setup/teardown trimming**: Derive constants via ALU doubling (2=1+1, 4=2+2, etc.) instead of const-loads. Skip loading indices (all start at 0). Skip storing indices (submission only checks values). Load only the 2 parameters actually needed.


**Increasing engine utilization**: Even without reducing total ops, we can get more done per cycle by using underutilized engines. ALU (12 slots/cycle) and flow (1 slot/cycle) are less pressured than VALU (6 slots/cycle) and load (2 slots/cycle), so moving work onto them is free throughput:

- **VALU→ALU promotion**: VALU ops that don't need native vector hardware (simple arithmetic) get decomposed into 8 scalar ALU ops on otherwise-idle ALU lanes. Partial promotion (e.g., 3/8 this cycle, 5/8 next) squeezes out remaining ALU capacity.
- **ALU→flow promotion**: Scalar address adds use the flow engine's `add_imm` instruction, freeing ALU slots for more vector promotions.
- **VALU slot swapping**: Non-promotable ops (multiply_add, vbroadcast) that need native VALU slots steal them from promotable ops already in the bundle, which then get demoted to ALU instead.

### 2. Scheduling heuristics

The core mental model: VALU ops produce work that feeds into loads and flow ops (gathers, vselects), which in turn unlock more VALU ops. Keeping this cycle saturated and never starving VALU of inputs is the goal.

The scheduler evolved through several heuristic iterations, guided by examining traces and utilization graphs:

- **FIFO**: Just prioritize based on what came first in the flat op list.
- **Distance to bottleneck**: Prioritize ops closest to unlocking a load or flow op (BFS distance in the DAG).
- **Adaptive resource bias**: Switch between load-biased and flow-biased scheduling based on how many of each are ready. If loads are running low, prioritize ops that feed loads.
- **Remaining-predecessors momentum**: The final heuristic uses the count of unscheduled predecessors feeding each bottleneck op. As ops get scheduled, the count drops, creating momentum toward finishing groups rather than spreading work thin. It also adaptively biases toward loads vs. flows based on how many of each are currently ready: early rounds (k=0-3) use mux trees which produce vselects (flow ops) but few loads, so we bias toward feeding the scarce loads to keep them saturated. Later rounds (k=4+) are all gathers, flooding the ready queue with loads, so we shift bias toward feeding flow ops instead.

The scheduler isn't optimal, but as long as VALU utilization stays close to 100%, the scheduler's choices are kind of good enough. Most improvements came from building telemetry (per-cycle slot utilization charts, per-vector readiness graphs, round completion timelines) and using it to identify specific bottleneck patterns, then tweaking heuristics to address them.

### 3. Mux trees vs. gathers

Gathering 8 values from non-contiguous memory requires 8 load_offset ops (4 cycles at 2 loads/cycle). For small tree levels, an alternative: preload all node values at that level, broadcast each into a vector, then use a binary mux tree of vselect ops to pick the right one per element.

This trades load slots (scarce, 2/cycle) for flow slots (vselect, 1/cycle) and VALU ops (broadcasts, bit extraction). For levels 0-3 (1 to 8 nodes), the trade is clearly worth it. For deeper levels, the vselect chain grows too long (and thus the number of additional valu/loads/broadcasts) and gathers win.

The mux/gather split ratios were tuned by examining slot utilization in the telemetry charts and experimenting with different numbers. Broadcast results are cached across rounds since tree structure doesn't change.

### 4. Allocator

The allocator uses a simple strategy: vectors scan left-to-right for contiguous 8-word blocks, scalars scan right-to-left for single words. This heuristic does an okay job avoiding fragmentation between the two size classes.

Key fixes along the way:
- Integrated allocation with scheduling so scratch pressure naturally throttles parallelism.
- Allowed overwriting a vreg's scratch space on the same cycle it's last read, enabling in-place reuse.
- Scratch pressure problems were usually symptoms of bad scheduling (spreading across too many vectors instead of finishing rounds), so fixing the scheduler often fixed the allocator pressure for free. Because of this, we didn't invest much effort in a smarter allocation strategy — the simple first-fit approach was sufficient once scheduling was reasonable.

## Tools

Claude was used extensively throughout -- for initial vectorization scaffolding, detailed operation-by-operation analysis of the hot loop, and implementing mechanical changes. The optimization ideas came from examining traces and utilization data, asking questions about why specific operations were needed, and iterating on scheduling heuristics based on observed bottleneck patterns.
