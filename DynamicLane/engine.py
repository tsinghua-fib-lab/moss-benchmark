from moss import Engine, LaneChange, TlPolicy, Verbosity


def get_engine(map_file, agent_file, start_step):
    eng = Engine(
        map_file=map_file,
        agent_file=agent_file,
        start_step=start_step,
        verbose_level=Verbosity.NO_OUTPUT,
        lane_change=LaneChange.MOBIL,
        lane_veh_add_buffer_size=1400,
        lane_veh_remove_buffer_size=1000,
    )
    eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.FIXED_TIME)
    eng.set_tl_duration_batch(range(eng.junction_count), 30)
    return eng
