from moss import Engine, TlPolicy, Verbosity


def get_engine(map_file, person_file, start_step,device):
    eng = Engine(
        name="TidalLane",
        map_file=map_file,
        person_file=person_file,
        start_step=start_step,
        verbose_level=Verbosity.NO_OUTPUT,
        device = device,
    )
    eng.set_tl_policy_batch([i for i in range(eng.junction_count)], TlPolicy.FIXED_TIME)
    eng.set_tl_duration_batch([i for i in range(eng.junction_count)], 30)
    return eng
