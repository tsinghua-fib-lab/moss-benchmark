from moss import Engine, TlPolicy, Verbosity


def get_engine(map_file, person_file, start_step, device):
    eng = Engine(
        name="TSC",
        map_file=map_file,
        person_file=person_file,
        start_step=start_step,
        device=device,
        verbose_level=Verbosity.NO_OUTPUT,
    )
    eng.set_tl_policy_batch([i for i in range(eng.junction_count)], TlPolicy.MANUAL)
    return eng
