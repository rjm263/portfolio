def combine_rules(*rules):
    def mask(D):
        m = np.ones(len(D), dtype=bool)
        for r in rules:
            if r is not None:
                m &= r(D)
        return m
    return mask

DATE_RULES = {'no_fridays': lambda d: d.weekday != 4,
                 'only_mondays': lambda d: d.weekday == 0,
                 'no_last_friday_of_month': lambda d: d.weekday != 4 and d.week == 4 if d.month != 2 else d.week == 3}