def eval_monotonic(alphas, losses):
    steps = len(alphas)
    assert steps == len(losses)

    increasing = False
    first_desc_idx = 0
    last_desc_idx = first_desc_idx

    bump_widths = []
    bump_heights = []
    active_bump_height = 0

    for i in range(1, steps):
        loss = losses[i]

        if not increasing:
            if loss > losses[i - 1]:
                increasing = True
                first_desc_idx = i - 1
                active_bump_height = loss - losses[i - 1]
        else:
            if loss < losses[first_desc_idx]:
                increasing = False
                last_desc_idx = i
                bump_widths.append(alphas[last_desc_idx] - alphas[first_desc_idx])
                bump_heights.append(active_bump_height)
            else:
                bump_height = losses[i] - losses[first_desc_idx]
                if bump_height > active_bump_height:
                    active_bump_height = bump_height

                if i == steps - 1:
                    last_desc_idx = i
                    bump_widths.append(alphas[last_desc_idx] - alphas[first_desc_idx])
                    bump_heights.append(active_bump_height)

    return bump_widths, bump_heights
