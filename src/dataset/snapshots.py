# coded attempt -> per step graph snapshot, each step is a explore_operation call
import copy
import networkx as nx
from dataclasses import dataclass, field
from src.preproc.reasoning_graph import GraphBuilder
from src.preproc.utils import run_code

GOAL = (24,)

@dataclass
class Snapshot:
    step: int # k: graph after the k-th operation
    graph: nx.DiGraph # ind deep copy
    current_state: tuple # state the last operation produced
    event: int 
    time_to_event: int # steps until solve (event = 1) or until last observed step (event = 0)
    # solved: event = 1, time_to_event = solve_step - k (>= 1) 
    # timed out: event = 0, time_to_event = n_ops - k (censored, 0 means no later step seen)

    n_backtracks: int = 0 # backtrack counts
    n_subgoals: int = 0 # subgoals counts

@dataclass
class AttemptSnapshots:
    snapshots: list = field(default_factory=list)
    solved: bool = False
    n_ops: int = 0 # explore_operation calls in the whole attempt
    solve_step: int = None # step that reaches 24, or last step + 1 if never stated
    goal_in_graph: bool = False # coded graph w/ edge into 24
    flags: list = field(default_factory=list) 

# replay GraphBuilder recorded actions, want to copy graph after each explore operation
def _replay(actions):
    # (action, builder) after each recorded action, rebuild graph
    builder = None
    for a in actions:
        t = a["type"]
        if t == "start":
            builder = GraphBuilder(a["state"])
        elif t == "explore_operation":
            builder.explore_operation(
                a["curr_state"], a["operation"], a["resulting_state"],
                result_calc_error=a["result_calc_error"], comment=a["comment"],
            )
        elif t == "move_to_node":
            builder.move_to_node(a["new_state"])
        elif t == "set_subgoal":
            builder.set_subgoal(a["subgoal_state"], a["state_after_subgoal"], a["comment"])
        yield a, builder

# make sure stop before op that reaches 24 so answer not in input
def snapshots_for_attempt(code: str, correct: float):
    # AttemptSnapshots, none if coder code doesn't run
    g = run_code(code, for_pretraining=False)
    if not isinstance(g, GraphBuilder):
        return None

    ops = [a for a in g.actions if a["type"] == "explore_operation"]
    n_ops = len(ops)
    solved = bool(correct == 1)
    reach = [i + 1 for i, a in enumerate(ops) if tuple(a["resulting_state"]) == GOAL]
    out = AttemptSnapshots(solved=solved, n_ops=n_ops, goal_in_graph=bool(reach))

    if solved:
        # first op reaching 24, if final step not spoken, it's just after the last step
        out.solve_step = reach[0] if reach else n_ops + 1
        if not reach:
            out.flags.append("solved_but_24_never_stated")
        last_input = out.solve_step - 1
    else:
        if reach:
            out.flags.append("unsolved_but_graph_reaches_24")
        last_input = n_ops
        # coder graph reaching 24 even if not solved, just stop before that step
        if reach:
            last_input = reach[0] - 1

    k = 0
    n_back = n_sub = 0
    position = None # current location
    for a, builder in _replay(g.actions):
        if a["type"] == "start":
            position = tuple(sorted(a["state"]))
            continue
        if a["type"] == "move_to_node":
            if tuple(a["new_state"]) != position:
                n_back += 1
            position = tuple(a["new_state"])
            continue
        if a["type"] == "set_subgoal":
            n_sub += 1
            continue
        if a["type"] != "explore_operation":
            continue
        position = tuple(a["resulting_state"])
        k += 1
        if k > last_input:
            break
        gc = copy.deepcopy(builder.G)
        if solved:
            event, tte = 1, out.solve_step - k
        else:
            event, tte = 0, n_ops - k
        out.snapshots.append(Snapshot(k, gc, tuple(a["resulting_state"]), event, tte, n_back, n_sub))
    return out