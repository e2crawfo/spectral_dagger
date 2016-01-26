from spectral_dagger import make_print_hook


def make_test_display(delay):
    if delay is None:
        return None
    else:
        delay = float(delay)
        return make_print_hook(delay)
