"""Shared bounded SIREN search design for standalone and recovery workflows."""


SIREN_LEARNING_RATES = (5e-5, 1e-4, 3e-4)


def siren_coordinate_mode(data_type=None, shape=None):
    if data_type == "audio":
        return "audio"
    if data_type in {"Video", "video"} or (shape is not None and len(shape) == 4):
        return "video"
    return "image"


def siren_tuning_candidates(trial_count, coordinate_mode="image", learning_rates=SIREN_LEARNING_RATES):
    """Sample the configured rates and network sizes within a bounded trial budget."""
    if not 1 <= trial_count <= 4:
        raise ValueError("SIREN trial_count must be in [1, 4]")
    if not learning_rates or any(isinstance(rate, bool) or not 1e-5 <= rate <= 1 for rate in learning_rates):
        raise ValueError("invalid SIREN learning-rate candidates")
    configurations = [(128, 3, 30.0, 30.0), (128, 3, 30.0, 30.0),
                      (256, 3, 60.0, 30.0), (256, 4, 20.0, 20.0)]
    # Start with the conventional rate when present, then cover both sides.
    rates = list(dict.fromkeys(learning_rates))
    if 1e-4 in rates:
        rates.remove(1e-4)
        rates.insert(0, 1e-4)
    return [{"hyperparameters": {"hidden_features": width, "hidden_layers": depth,
                                  "first_omega_0": first, "hidden_omega_0": hidden,
                                  "coordinate_mode": coordinate_mode},
             "learning_rate": rates[index % len(rates)]}
            for index, (width, depth, first, hidden) in enumerate(configurations[:trial_count])]
