INF = 1e18

def get_sc_prob(lap: int, sc_prob_ranges):
    for (mn, mx, p) in sc_prob_ranges:
        if mx is None and lap >= mn:
            return p
        if mn <= lap <= mx:
            return p
    raise RuntimeError(f"No SC probability defined for lap {lap}")

def lap_time(comp_idx, age, lap, compounds, fuel_effect):
    base = compounds[comp_idx]['pace']
    deg = compounds[comp_idx]['degradation']
    return base + deg * age - fuel_effect * (lap - 1)

class TwoCompoundTyreRule:
    # meta=the one compound already used, or meta=-1 if at least two compounds were already used
    def __init__(self, compound_indices):
        self.compound_indices = compound_indices

    def all_meta_new(self):
        ans = [(-1,list(self.compound_indices))]
        ans += [(c,[c]) for c in self.compound_indices]
        return ans

    def init_meta(self, start_comp_idx):
        return start_comp_idx

    def new_compounds_allowed(self, meta):
        return list(self.compound_indices)

    def next_meta_on_sc(self, meta):
        return meta  # no change on SC

    def next_meta_on_pit(self, meta, new_comp):
        if meta != new_comp: return -1
        else: return meta

    def terminal_ok(self, meta):
        return meta == -1 # At least two different compounds used.

class PatternUnlessSCTyreRule:
    # before SC: meta=(False, index) of the next compound, meta=length means we used all
    # after SC:  meta=(True, index) of one compound already used, or index=-1 if at least two compounds were already used
    def __init__(self, pattern_comp_indices, compound_indices):
        self.pattern = pattern_comp_indices
        self.length = len(self.pattern)
        self.compound_indices = compound_indices

    def all_meta_new(self):
        metas = [((False,i), [self.pattern[i-1]])  for i in range(1, self.length + 1)]
        metas.append(((True, -1), list(self.compound_indices)))
        metas += [((True, c), [c] ) for c in self.compound_indices]
        return metas

    def init_meta(self, start_comp_idx):
        if start_comp_idx != self.pattern[0]: return None
        return (False, 1)

    def new_compounds_allowed(self, meta):
        sc_flag, idx = meta
        if sc_flag: 
            return list(self.compound_indices)
        else:
            if idx >= self.length: return []
            return [self.pattern[idx]]

    def next_meta_on_pit(self, meta, new_comp):
        sc_flag, idx = meta

        if sc_flag:
            if idx == -1: return (True, -1)
            if new_comp == idx: return (True, idx)
            else: return (True, -1)
        else:
            if idx >= self.length or new_comp != self.pattern[idx]: return None
            return (False, idx + 1)

    def next_meta_on_sc(self, meta):
        sc_flag, idx = meta
        if sc_flag: 
            return meta
        else:
            used = set(self.pattern[:idx])
            if len(used) >= 2: return (True, -1)
            else: return (True, self.pattern[idx-1])

    def terminal_ok(self, meta):
        sc_flag, idx = meta
        if sc_flag: 
            return idx == -1
        else:
            return idx == self.length

def compute_policy(num_laps, compounds, fuel_effect, sc_prob_ranges, sc_length, pit_loss, rule):
    sc_prob = {lap: get_sc_prob(lap, sc_prob_ranges) for lap in range(1, num_laps + 1)}
    dp_exp = {}
    policy = {}
    follow_states = {}

    for lap in range(num_laps + 1, 0, -1):
        for meta, meta_compounds in rule.all_meta_new():
            for comp_idx in meta_compounds:
                for sc_status in [0,1]: # 1 represents that SC has just started
                    for age in range(0, lap):
                        state = (lap, comp_idx, age, sc_status, meta)

                        # Terminal (beyond last lap)
                        if lap == num_laps + 1:
                            dp_exp[state] = 0 if rule.terminal_ok(meta) else INF
                            policy[state] = None
                            follow_states[state] = None
                            continue

                        best_exp = INF
                        best_action = None
                        best_follow = None

                        if sc_status == 1:
                            # Safety Car: no lap time cost, free pit on the first SC lap, then couunt down sc_length-1 laps
                            options = [(comp_idx, age, ("stay",), meta, 0)]
                            for new_c in rule.new_compounds_allowed(meta):
                                options.append((new_c, 0, ("pit", new_c), rule.next_meta_on_pit(meta, new_c), 0)) # pit_loss is zero

                            for option in options:
                                comp_idx_opt, age_opt, action_opt, meta_opt, pit_loss_opt = option
                                next_state = (min(lap + sc_length - 1, num_laps + 1), comp_idx_opt, age_opt, 0, rule.next_meta_on_sc(meta_opt))
                                exp = pit_loss_opt + dp_exp[next_state]
                                if exp < best_exp:
                                    best_exp = exp
                                    best_action = action_opt
                                    best_follow = [next_state]

                        else:
                            # Green-flag lap
                            options = [(comp_idx, age, ("stay",), meta, 0)]
                            for new_c in rule.new_compounds_allowed(meta):
                                options.append((new_c, 0, ("pit", new_c), rule.next_meta_on_pit(meta, new_c), pit_loss))

                            for option in options:
                                comp_idx_opt, age_opt, action_opt, meta_opt, pit_loss_opt = option
                                lt = lap_time(comp_idx_opt, age_opt, lap, compounds, fuel_effect)
                                next_sc_state = (lap + 1, comp_idx_opt, age_opt + 1, 1, rule.next_meta_on_sc(meta_opt))
                                val_sc = dp_exp[next_sc_state]
                                next_green_state = (lap + 1, comp_idx_opt, age_opt + 1, 0, meta_opt)
                                val_green = dp_exp[next_green_state]
                                exp = pit_loss_opt + sc_prob[lap] * val_sc + (1 - sc_prob[lap]) * (val_green + lt)
                                if exp < best_exp:
                                    best_exp = exp
                                    best_action = action_opt
                                    best_follow = [next_sc_state, next_green_state]

                        dp_exp[state] = best_exp
                        policy[state] = best_action
                        follow_states[state] = best_follow

    return dp_exp, policy, follow_states

def extract_no_sc_strategy(rule, start_comp_idx, num_laps, dp_policy, compounds):
    meta = rule.init_meta(start_comp_idx)
    
    stints = []
    comp_idx = start_comp_idx
    age = 0
    current_stint_start = 1

    for lap in range(1, num_laps + 1):
        state = (lap, comp_idx, age, 0, meta)
        action = dp_policy[state]
        if action[0] == 'pit':
            new_comp = action[1]
            stints.append((comp_idx, current_stint_start, lap - 1))
            new_meta = rule.next_meta_on_pit(meta, new_comp)
            comp_idx = new_comp
            meta = new_meta
            current_stint_start = lap
            age = 1
        else:
            age += 1

    if current_stint_start <= num_laps:
        stints.append((comp_idx, current_stint_start, num_laps))

    return stints

def calc_race_time(stints, compounds, fuel_effect, pit_loss):
    total_time = pit_loss * (len(stints) - 1)
    for ci, s_lap, e_lap in stints:
        for lap in range(s_lap, e_lap + 1):
            total_time += lap_time(ci, lap - s_lap, lap, compounds, fuel_effect)
    return total_time

def summarize_results(num_laps, compounds, fuel_effect, sc_prob_ranges, sc_length, pit_loss, rule):
    dp_exp, policy, _ = compute_policy(num_laps, compounds, fuel_effect, sc_prob_ranges, sc_length, pit_loss, rule)    
    results = {}
    for start_idx, compound in enumerate(compounds):
        meta0 = rule.init_meta(start_idx)
        if meta0 is None: continue  # starting on this tyre is illegal under this rule
        stints = extract_no_sc_strategy(rule, start_idx, num_laps, policy, compounds)
        expected_time = dp_exp[(1, start_idx, 0, 0, meta0)]
        results[compound['type']] = {
            "expected_time": expected_time,
            "no_sc_stints": stints,
            "no_sc_time": calc_race_time(stints, compounds, fuel_effect, pit_loss),
        }
    return results
    
def main():
    num_laps = 57
    compounds = [
        {"type": "H", "pace": 83.559, "degradation": 0.08},
        {"type": "M", "pace": 83.167, "degradation": 0.09},
        {"type": "S", "pace": 83.78, "degradation": 0.22},
    ]
    fuel_effect = 0.08
    # sc_prob_ranges = [
    #     (1, 1, 0.261),
    #     (2, 2, 0.049),
    #     (3, 3, 0.033),
    #     (4, None, 0.016),
    # ]
    sc_prob_ranges = [
        (1, 1, 0.261),
        (2, 2, 0.049),
        (3, 3, 0.033),
        (4, None, 0.016),
    ]
    sc_length = 5
    pit_loss = 27.824

    type_to_idx = {c["type"]: i for i, c in enumerate(compounds)}
    rule = TwoCompoundTyreRule(compound_indices=list(range(len(compounds))))
    results = summarize_results(num_laps, compounds, fuel_effect, sc_prob_ranges, sc_length, pit_loss, rule)
    print("=== Unconstrained rule (≥2 compounds) ===")
    for start_comp, info in results.items():
        print(f"Start on {start_comp}:")
        print(f"  Expected time: {info['expected_time']}")
        print(f"  No-SC optimal stints: {info['no_sc_stints']}")
        print(f"  No-SC race time: {info['no_sc_time']}")
        print()    

    for pattern in [["M", "H"], ["M", "S"], ["H", "M"], ["M", "H", "H"], ["H", "M", "M"]]:
        pattern_indices = [type_to_idx[t] for t in pattern]
        rule = PatternUnlessSCTyreRule(pattern_indices, compound_indices=list(range(len(compounds))))
        results = summarize_results(num_laps, compounds, fuel_effect, sc_prob_ranges, sc_length, pit_loss, rule)
        print(f"=== Pattern unless SC {', '.join(pattern)} ===")
        for start_comp, info in results.items():
            print(f"  Expected time: {info['expected_time']}")
            print(f"  No-SC optimal stints: {info['no_sc_stints']}")
            print(f"  No-SC race time: {info['no_sc_time']}")
            print()
    
main()