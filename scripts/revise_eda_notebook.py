"""Build the consolidated A1 draft; run the resulting notebook separately."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(source):
    cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(dedent(source).strip()))


markdown("""
    # Chicago Food Inspections: Assessment 1 EDA

    Predict whether an inspection will **Fail**, using information available
    before that inspection begins. The analysis uses a fixed local snapshot.
    Run from the project root or from `base/` with the supplied `data/` folder.

    **Analysis boundaries.** Raw-data provenance, missingness and coverage are
    audited over the saved snapshot. Outcome comparisons and historical-feature
    exploration use inspections before **2025-01-01**. The evaluation plan below
    records limitations from prior exploration of later outcomes. No model is
    trained here, and official hidden-period outcomes are never requested or used.
""")

markdown("""
    ## 1. Load and verify the saved snapshot

    A row is one inspection, not one establishment. `Inspection ID` identifies
    the visit; `License #` is an imperfect establishment identity key. Read licence
    numbers and ZIP codes as strings. A requested end date in a filename is not
    evidence that the snapshot contains records on that date.

    The repository stores the raw CSV losslessly as `.csv.gz`. It is read
    directly, without manual extraction or downloads. A local uncompressed copy
    of the same snapshot is also supported. The SHA-256 check below always
    verifies the decompressed CSV bytes against the original downloaded file.
""")

code("""
    from pathlib import Path
    import ast
    import gzip
    import hashlib
    import json
    import platform

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    SNAPSHOT_FILE = 'food_inspections_2016-01-01_2026-10-25.csv'
    EXPECTED_SHA256 = '6ebba026f5b8f6c21c2b87efac548b14d0dff764175b706482b46e71e0b436ca'
    TRAIN_END = pd.Timestamp('2024-12-31')
    VALID_START = pd.Timestamp('2025-01-01')
    TEST_START = pd.Timestamp('2026-01-01')
    INTERNAL_END = pd.Timestamp('2026-09-30')
    LABEL_MAP = {'Fail': 1, 'Pass': 0, 'Pass w/ Conditions': 0}

    candidates = [Path.cwd(), Path.cwd().parent]
    PROJECT_ROOT = next((p for p in candidates if any(
        (p / 'data' / name).is_file() for name in [SNAPSHOT_FILE, SNAPSHOT_FILE + '.gz']
    )), None)
    assert PROJECT_ROOT is not None, 'Run from the project root or base/ with the saved data/ folder.'
    DATA_PATH = PROJECT_ROOT / 'data' / SNAPSHOT_FILE
    if not DATA_PATH.is_file():
        DATA_PATH = DATA_PATH.with_suffix('.csv.gz')
    META_PATH = PROJECT_ROOT / 'data' / 'download_metadata.json'
    snapshot_open = gzip.open if DATA_PATH.suffix == '.gz' else open
    with snapshot_open(DATA_PATH, 'rb') as stream:
        snapshot_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert snapshot_hash == EXPECTED_SHA256, 'Snapshot changed: review the version before updating this notebook.'

    raw = pd.read_csv(DATA_PATH, dtype={'License #': 'string', 'Zip': 'string'}, low_memory=False)
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    assert meta['csv_file'] == SNAPSHOT_FILE
    assert meta['row_count'] == len(raw)
    assert raw.shape[1] == 17

    df = raw.copy()
    df['Inspection Date'] = pd.to_datetime(raw['Inspection Date'], format='%m/%d/%Y', errors='coerce')
    invalid_date_count = int(df['Inspection Date'].isna().sum())
    assert invalid_date_count == 0, 'Investigate invalid dates before time-based analysis.'
    assert df['Inspection Date'].between('2016-01-01', '2026-10-30').all()
    assert df['Inspection Date'].between(meta['requested_window']['start_date'], meta['requested_window']['end_date']).all()
    exact_duplicates = int(raw.duplicated().sum())
    duplicate_ids = int(raw['Inspection ID'].duplicated().sum())
    assert raw['Inspection ID'].notna().all() and duplicate_ids == 0
    assert exact_duplicates == 0

    df['fail'] = df['Results'].map(LABEL_MAP).astype('Int64')
    labelled = df.loc[df['fail'].notna()].copy()
    assert df['fail'].notna().equals(df['Results'].isin(LABEL_MAP))
    assert labelled['fail'].eq(labelled['Results'].eq('Fail').astype(int)).all()

    provenance = pd.DataFrame([
        ['Source', meta['portal_page']],
        ['Downloaded (UTC)', meta['downloaded_at_utc']],
        ['Requested window', str(meta['requested_window'])],
        ['Actual first inspection', str(df['Inspection Date'].min().date())],
        ['Actual last inspection', str(df['Inspection Date'].max().date())],
        ['Raw rows / columns', f'{raw.shape[0]:,} / {raw.shape[1]}'],
        ['Labelled / unlabelled rows', f'{len(labelled):,} / {df.fail.isna().sum():,}'],
        ['SHA-256', snapshot_hash],
    ], columns=['item', 'value'])
    with pd.option_context('display.max_colwidth', None):
        display(provenance)
    print('Python:', platform.python_version(), '| pandas:', pd.__version__, '| numpy:', np.__version__)
    display(raw.head(3))

    plt.rcParams.update({
        'figure.dpi': 110, 'font.size': 10, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    BLUE, RED, GREEN, GREY = '#3274A1', '#C44E52', '#33856A', '#777777'
""")

markdown("""
    ## 2. Missing values are not the same as invalid identifiers

    `isna()` detects empty/NA values, not the literal number or string `0`.
    A missing percentage rounded to `0.00%` can still represent several records.
    Report both counts and sufficiently precise rates. The source does not tell
    us why zero licences were entered, so the decision below is a conservative
    identity policy, not a claim about the portal's official coding convention.

    **Official source check:** the portal-linked data dictionary defines licence
    numbers as unique licensing identifiers but assigns no explicit meaning to
    zero. A saved copy is in `data/documentation/food_inspections_data_dictionary.pdf`;
    the online verification record is saved separately. Source checks occurred
    during preparation, not during this offline notebook run.

    **Policy:** retain all visits and eligible labels. Use only positive,
    integer-form licence strings for linkage. Missing, zero or malformed licences
    have **unknown identity**, unavailable history and a separate flag. Do not
    group all zero/missing values together; do not infer aliases from names or
    addresses without validating a record-linkage method.
""")

code("""
    missing = pd.DataFrame({
        'missing_count': raw.isna().sum(),
        'missing_percent': 100 * raw.isna().mean(),
        'dtype': raw.dtypes.astype(str),
    }).sort_values('missing_count', ascending=False)
    display(missing.style.format({'missing_percent': '{:.4f}'}))

    licence = raw['License #'].str.strip()
    licence_number = pd.to_numeric(licence, errors='coerce')
    missing_licence = licence.isna() | licence.eq('').fillna(False)
    zero_licence = licence_number.eq(0).fillna(False) & ~missing_licence
    valid_licence = (licence.str.fullmatch(r'[0-9]+').fillna(False) & licence_number.gt(0).fillna(False))
    other_invalid = ~(missing_licence | zero_licence | valid_licence)
    licence_status = pd.Series('valid_positive', index=df.index, dtype='string')
    licence_status.loc[missing_licence] = 'missing'
    licence_status.loc[zero_licence] = 'zero'
    licence_status.loc[other_invalid] = 'other_invalid'
    df['licence_status'] = licence_status
    df['identity_unknown'] = ~valid_licence
    df['establishment_key'] = licence.where(valid_licence)

    licence_audit = df.groupby('licence_status', observed=True).agg(
        records=('Inspection ID', 'size'),
        distinct_names=('DBA Name', 'nunique'),
        distinct_addresses=('Address', 'nunique'),
    ).reindex(['valid_positive', 'missing', 'zero', 'other_invalid'], fill_value=0)
    licence_audit['percent_of_all_records'] = 100 * licence_audit['records'] / len(df)
    display(licence_audit.round(4))
    print('Raw licence NA percentage rounded to two decimals:', round(100 * raw['License #'].isna().mean(), 2))
    display(df.loc[zero_licence, ['Inspection ID', 'DBA Name', 'License #', 'Address', 'Results']].head(6))
    assert df.loc[df.identity_unknown, 'establishment_key'].isna().all()
    assert not df.loc[~df.identity_unknown, 'establishment_key'].eq('0').any()
    assert licence_audit['records'].sum() == len(df)
""")

markdown("""
    ### Categories and geography: verify before simplifying

    Apply fixed whitespace/case rules to facility and inspection types, without
    merging different business meanings. Preserve raw fields. Frequency-based
    category selection will use the training period only. City has both spelling
    variants and genuinely different city names: not every value is a Chicago typo.

    Address is street/house-number text; ZIP is a postal area, not a community
    boundary. `Location` is a coordinate pair in this CSV. We check that it agrees
    with latitude and longitude before calling it redundant. Agreement does not
    establish geocoding accuracy. No community boundary file is introduced here.
""")

code("""
    def normalise_category(series):
        return (series.astype('string').str.strip().str.replace(r'\\s+', ' ', regex=True)
                .str.casefold().replace('', pd.NA).fillna('Unknown'))

    df['facility'] = normalise_category(df['Facility Type'])
    df['inspection_type'] = normalise_category(df['Inspection Type'])
    df['risk'] = df['Risk'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
    df['zip_category'] = df['Zip'].str.strip().replace('', pd.NA).fillna('Unknown')
    category_audit = pd.DataFrame({
        'raw_nonmissing_categories': [raw[c].nunique(dropna=True) for c in ['Facility Type', 'Inspection Type']],
        'normalised_nonmissing_categories': [
            df.loc[raw['Facility Type'].notna(), 'facility'].nunique(),
            df.loc[raw['Inspection Type'].notna(), 'inspection_type'].nunique(),
        ],
    }, index=['Facility Type', 'Inspection Type'])
    display(category_audit)
    display(raw['City'].value_counts(dropna=False).head(12).to_frame('records'))

    parsed_locations = {}
    for text in raw['Location'].dropna().unique():
        pair = ast.literal_eval(text)
        assert isinstance(pair, (tuple, list)) and len(pair) == 2
        parsed_locations[text] = (float(pair[0]), float(pair[1]))
    pairs = raw['Location'].map(parsed_locations)
    complete_coords = raw[['Latitude', 'Longitude', 'Location']].notna().all(axis=1)
    lat_match = np.isclose(pairs.str[0][complete_coords].astype(float), raw.loc[complete_coords, 'Latitude'], atol=1e-9, rtol=0)
    lon_match = np.isclose(pairs.str[1][complete_coords].astype(float), raw.loc[complete_coords, 'Longitude'], atol=1e-9, rtol=0)
    coordinate_mismatches = int((~(lat_match & lon_match)).sum())
    matching_coordinate_gaps = (raw.Latitude.isna().equals(raw.Longitude.isna())
                                and raw.Latitude.isna().equals(raw.Location.isna()))
    assert coordinate_mismatches == 0 and matching_coordinate_gaps
    print(f'Complete coordinate rows: {complete_coords.sum():,}; mismatches: {coordinate_mismatches}')
    with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
        display(raw[['Address', 'City', 'State', 'Zip', 'Latitude', 'Longitude', 'Location']].head(3))
        display(raw.loc[raw.Latitude.isna(), ['Address', 'Zip', 'Latitude', 'Longitude']].head(3))

    train_records = df.loc[df['Inspection Date'] <= TRAIN_END].copy()
    eda = train_records.loc[train_records.fail.notna()].copy()
    eda['fail'] = eda['fail'].astype(int)
    training_fail_rate = eda.fail.mean()
    print(f'Training-period EDA: {len(eda):,} labelled visits; Fail rate {training_fail_rate:.2%}')
""")

markdown("""
    ## 3. EDA: eight figures tied to modelling questions

    Dashed reference lines in the group-rate charts mark the overall labelled
    training-period failure rate; each group's denominator is its labelled visits.

    ### Figure 1. Which outcomes become labels?

    **Comparison:** training-period counts of all recorded outcomes. Colours
    distinguish Fail, non-Fail and excluded outcomes. **Finding:** non-Fail is
    more frequent than Fail; operational outcomes are not negative labels.
    **Implication:** accuracy alone can hide missed failures; use the specified
    Macro-F1 and retain earlier unlabelled visits as separate historical evidence.
    **Limit:** inspection counts are not counts of unique establishments.
""")

code("""
    results = train_records['Results'].value_counts(dropna=False).sort_values()
    result_table = results.rename('n').to_frame()
    result_table['project_label'] = result_table.index.map(LABEL_MAP)
    display(result_table)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    colours = [RED if LABEL_MAP.get(r) == 1 else BLUE if LABEL_MAP.get(r) == 0 else GREY for r in results.index]
    ax.barh(results.index.astype(str), results.values, color=colours)
    for y, n in enumerate(results.values):
        ax.text(n + results.max() * .015, y, f'{n:,}', va='center', fontsize=9)
    ax.set(xlabel='Number of inspections', ylabel='Recorded outcome',
           title='Figure 1. Recorded outcomes in the training period', xlim=(0, results.max() * 1.2))
    fig.tight_layout()
    plt.show()
    print(f'Fail prevalence among labelled training visits: {training_fail_rate:.2%}')
""")

markdown("""
    ### Figure 2. Is time coverage stable?

    **Comparison:** all raw monthly visit counts, plus annual failure rates only
    in the training period. Months without records are explicitly filled with zero.
    **Finding:** counts and training-period rates vary over time; the final month
    is only partly observed in this snapshot. **Implication:** preserve chronology,
    and do not compare a partial year/month to complete periods as if exposure were
    equal. **Limit:** counts alone cannot prove a reporting-lag duration or the cause
    of a drop. Future-period labels are not plotted here.

    The official portal flags a **2018-07-01 inspection-procedure change**. Treat
    this as a possible comparability issue across years, not proof that it caused
    any particular movement in failure rates.
""")

code("""
    monthly = df.groupby(df['Inspection Date'].dt.to_period('M')).size()
    month_index = pd.period_range(df['Inspection Date'].min(), df['Inspection Date'].max(), freq='M')
    monthly = monthly.reindex(month_index, fill_value=0)
    annual = eda.groupby(eda['Inspection Date'].dt.year).agg(n=('fail', 'size'), fail_rate=('fail', 'mean'))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.3))
    axes[0].plot(monthly.index.to_timestamp(), monthly.values, color=BLUE)
    axes[0].scatter(monthly.index[-1].to_timestamp(), monthly.iloc[-1], color=RED, zorder=3)
    axes[0].set(title='Figure 2a. Raw monthly coverage; final month is partial', ylabel='All recorded visits')
    axes[1].plot(annual.index, 100 * annual.fail_rate, marker='o', color=RED)
    axes[1].set(title='Figure 2b. Training-period annual failure rate', xlabel='Year', ylabel='Fail among labelled visits (%)')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    plt.show()
    display(annual.assign(fail_rate_percent=100 * annual.fail_rate).drop(columns='fail_rate').round(2))
    print('Final recorded month / raw visits:', str(monthly.index[-1]), int(monthly.iloc[-1]))
""")

markdown("""
    ### Figure 3. Does missing violation text depend on the recorded outcome?

    **Comparison:** the fraction with missing current-inspection Violations, within
    each training-period outcome. **Finding:** missingness differs substantially
    across outcomes. **Implication:** exclude the current text, its missing flag and
    any derived counts from predictors. **Limit:** association does not establish
    why text is absent. The exclusion is justified by the field's post-start timing,
    not merely by correlation. Historical violations would require separate timing
    and availability controls and are outside the proposed first baseline.
""")

code("""
    violation_gaps = train_records.assign(violation_missing=train_records['Violations'].isna()).groupby('Results').agg(
        n=('Inspection ID', 'size'), missing_rate=('violation_missing', 'mean')
    ).sort_values('missing_rate')
    display(violation_gaps.assign(missing_percent=100 * violation_gaps.missing_rate).drop(columns='missing_rate').round(2))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(violation_gaps.index, 100 * violation_gaps.missing_rate, color=GREY)
    for y, row in enumerate(violation_gaps.itertuples()):
        ax.text(100 * row.missing_rate + 1, y, f'{100 * row.missing_rate:.1f}% (n={row.n:,})', va='center', fontsize=8)
    ax.set(xlim=(0, 135), xlabel='Missing current Violations (%)', ylabel='Recorded outcome',
           title='Figure 3. Missingness is outcome-associated, not a usable pre-visit feature')
    fig.tight_layout()
    plt.show()
""")

markdown("""
    ### Figure 4. Does inspection type distinguish observed failures?

    **Comparison:** within-type failure rates among labelled training visits, using
    normalised categories with at least 200 visits. **Finding:** complaint, licence,
    routine and re-inspection groups have different observed rates. **Implication:**
    inspection type is a plausible pre-visit categorical feature. **Limit:** this
    is not a causal effect or proof that it is the strongest predictor; selection
    processes and history also differ. Display cutoffs are not automatic modelling
    category-merging rules.

    **Reading the bars and dashed line.** Each bar is the number of Fail visits
    divided by the number of labelled visits within that inspection type. The
    grey vertical dashed line is the overall **2016-2024 labelled-training Fail
    rate: 32,189 / 140,762 = 22.87%**. It includes all training inspection types,
    including those not plotted. Bars to its right have higher observed rates;
    bars to its left have lower observed rates. This is a pooled, visit-weighted
    reference, not the unweighted mean of the plotted bars, a prediction threshold
    or a statistical-significance boundary. It does not adjust for other factors.
    The n >= 200 cutoff is a display choice to reduce small-group volatility and
    crowding, not a course rule or a guarantee of precision. Excluded categories
    and their records remain in the dataset.
""")

code("""
    def rate_table(frame, column):
        table = frame.groupby(column, observed=True)['fail'].agg(n='size', fail_count='sum', fail_rate='mean')
        assert (table.fail_count <= table.n).all()
        return table

    def draw_rate_bars(ax, table, title, colour=BLUE, sort_by_rate=True):
        if sort_by_rate:
            table = table.sort_values('fail_rate')
        ax.barh(table.index.astype(str), 100 * table.fail_rate, color=colour)
        for y, row in enumerate(table.itertuples()):
            ax.text(100 * row.fail_rate + .5, y, f'{100 * row.fail_rate:.1f}% (n={row.n:,})', va='center', fontsize=8)
        ax.axvline(100 * training_fail_rate, color=GREY, linestyle='--', linewidth=1)
        ax.set(xlim=(0, max(100 * table.fail_rate.max() + 20, 45)), xlabel='Fail among labelled visits (%)', title=title)

    type_stats = rate_table(eda, 'inspection_type')
    shown_types = type_stats.loc[type_stats.n >= 200]
    display(shown_types.sort_values('fail_rate', ascending=False).round(4))
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    draw_rate_bars(ax, shown_types, 'Figure 4. Inspection type (training n >= 200)')
    fig.tight_layout()
    plt.show()
""")

markdown("""
    ### Figure 5. Risk grade and common facility types

    **Comparison:** named risk grades and the ten most frequent normalised facility
    types in the training period. The latter are selected by frequency, not by high
    failure rates. **Finding:** recorded risk grades are not a simple ordering of
    observed failure probability; facility groups vary. **Implication:** retain them
    as candidate categoricals; learn rare-level rules on training data. **Limit:**
    these comparisons do not establish the official mechanism behind risk grades
    or explain the differences. The official data dictionary ties Risk to adverse
    public-health risk and inspection frequency, not a calibrated Fail probability.
    Missing categories are retained even when not among the categories plotted.

    **Reading the bars and dashed lines.** A bar is the Fail proportion among
    labelled training visits in its risk or facility category; n counts visits,
    not unique businesses. Both grey dashed lines mark the same overall training
    reference, **32,189 / 140,762 = 22.87%**. The reference includes every labelled
    training visit, including unknown Risk and facility types outside the ten
    displayed. It is not recomputed separately for each panel and is not an
    unweighted mean of its bars. A bar beyond the line is above the pooled rate;
    a bar before it is below. This visual comparison establishes neither a
    statistically significant difference nor a causal effect, and the line is
    not a model decision threshold. Facility categories are selected by visit
    count, then ordered by observed failure rate for readability.
""")

code("""
    risk_order = ['Risk 1 (High)', 'Risk 2 (Medium)', 'Risk 3 (Low)']
    risk_stats = rate_table(eda, 'risk').reindex(risk_order)
    facility_stats = rate_table(eda, 'facility')
    common_facilities = facility_stats.nlargest(10, 'n')
    top10_share = common_facilities.n.sum() / len(eda)
    print(f'Ten most frequent facilities cover {top10_share:.2%} of labelled training visits.')
    display(risk_stats.round(4))
    display(common_facilities.sort_values('n', ascending=False).round(4))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    draw_rate_bars(axes[0], risk_stats, 'Figure 5a. Recorded risk grade', GREEN, sort_by_rate=False)
    draw_rate_bars(axes[1], common_facilities, 'Figure 5b. Ten common facility types')
    axes[1].tick_params(axis='y', labelsize=8)
    fig.tight_layout(w_pad=2)
    plt.show()
""")

markdown("""
    ### Figure 6. ZIP-level variation, with both ends of the distribution

    **Comparison:** highest and lowest eight failure rates among ZIP groups with at
    least 300 labelled training visits. The full eligible ZIP table is also shown.
    **Finding:** sufficiently represented ZIP groups differ. **Implication:** ZIP is
    a candidate area category, not an ordinal number. **Limit:** these are postal
    areas, not community boundaries; facility mix, inspection reasons and repeated
    establishments may explain part of the difference. This is not a causal ranking
    of neighbourhood hygiene. Coordinate value beyond ZIP remains untested.

    **Reading the bars and dashed lines.** Each ZIP rate divides its Fail visits
    by its labelled training visits. Both panels use the same 0-60% horizontal
    scale and the same grey dashed reference: **32,189 / 140,762 = 22.87%** across
    all labelled training visits. The reference includes visits with unknown ZIP
    and those in groups below the 300-visit display cutoff; it is not the mean
    of the selected high or low ZIP bars. Bars to the right of the reference are
    above the pooled rate and bars to the left are below. The line is neither
    a significance boundary nor a prediction threshold. Selecting extreme ZIP
    rates is descriptive, not an unbiased ranking of underlying area risk.
    The n >= 300 cutoff limits very small groups for display, does not guarantee
    precision and does not remove records from the dataset.
""")

code("""
    zip_stats = rate_table(eda, 'zip_category')
    zip_eligible = zip_stats.loc[(zip_stats.n >= 300) & (zip_stats.index != 'Unknown')]
    print(f'Eligible ZIP groups cover {zip_eligible.n.sum() / len(eda):.2%} of labelled training visits.')
    display(zip_eligible.sort_values('fail_rate', ascending=False).round(4))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    draw_rate_bars(axes[0], zip_eligible.nlargest(8, 'fail_rate'), 'Figure 6a. Higher observed ZIP rates')
    draw_rate_bars(axes[1], zip_eligible.nsmallest(8, 'fail_rate'), 'Figure 6b. Lower observed ZIP rates', GREEN)
    for ax in axes:
        ax.set_xlim(0, 60)
    fig.tight_layout()
    plt.show()
""")

markdown("""
    ### Figure 7. Strictly earlier-day history, without identity collisions

    Aggregate visits by **valid licence and date** first. Cumulative counts exclude
    the entire current day, so changing the order of same-day inspection IDs cannot
    reveal another current-day result. History counts all past visits; failure-rate
    denominators count only past labelled visits. Unknown identities remain in the
    labelled dataset but are excluded from these history-group comparisons.

    This is an EDA construction within the training period, not permission to update
    hidden-test history using hidden outcomes. With no dates before 2016 in our
    saved inputs, “no prior observed visit” does not mean “never inspected before”.

    **Reading the bars and dashed lines.** Each bar measures the current Fail
    proportion among labelled, valid-identity training visits in that history
    group. The bars do not show historical failure rates. Both panels retain
    the same overall training reference as Figures 4-6: **32,189 / 140,762 =
    22.87%**, including the 325 unknown-identity labelled visits omitted from
    these bars. The reference is deliberately not recalculated on the history
    subset. Positions relative to it describe above/below-overall observed rates,
    not significance, causation or a model threshold. Counts and intervals follow
    their natural order rather than being sorted by failure rate.
""")

code("""
    def add_prior_day_history(records):
        valid = records.loc[records.establishment_key.notna()].copy()
        valid['labelled_visit'] = valid.fail.notna().astype(int)
        valid['failed_visit'] = valid.fail.eq(1).fillna(False).astype(int)
        daily = valid.groupby(['establishment_key', 'Inspection Date'], as_index=False).agg(
            day_visits=('Inspection ID', 'size'), day_labelled=('labelled_visit', 'sum'),
            day_fails=('failed_visit', 'sum'),
        ).sort_values(['establishment_key', 'Inspection Date'])
        groups = daily.groupby('establishment_key', sort=False)
        for source, target in [('day_visits', 'prior_visits'), ('day_labelled', 'prior_labelled'), ('day_fails', 'prior_fails')]:
            daily[target] = groups[source].cumsum() - daily[source]
        daily['previous_visit_date'] = groups['Inspection Date'].shift()
        daily['days_since_previous'] = (daily['Inspection Date'] - daily['previous_visit_date']).dt.days
        daily['prior_fail_rate'] = daily['prior_fails'] / daily['prior_labelled'].replace(0, np.nan)
        history_columns = ['prior_visits', 'prior_labelled', 'prior_fails', 'previous_visit_date', 'days_since_previous', 'prior_fail_rate']
        return records.merge(daily[['establishment_key', 'Inspection Date'] + history_columns],
                             on=['establishment_key', 'Inspection Date'], how='left', validate='many_to_one')

    # Same-day results, unlabelled visits and invalid identities exercise different rules.
    example = pd.DataFrame({
        'Inspection ID': [1, 2, 3, 4, 5, 6],
        'establishment_key': ['A', 'A', 'A', 'A', pd.NA, pd.NA],
        'Inspection Date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-02', '2024-01-03']),
        'fail': pd.array([1, 0, pd.NA, 0, 1, 0], dtype='Int64'),
    })
    checked = add_prior_day_history(example).set_index('Inspection ID')
    assert checked.loc[[1, 2], 'prior_visits'].eq(0).all()
    assert checked.loc[3, 'prior_fails'] == 1 and checked.loc[3, 'prior_labelled'] == 2
    assert checked.loc[4, 'prior_visits'] == 3 and checked.loc[4, 'prior_fail_rate'] == .5
    assert checked.loc[[5, 6], 'prior_visits'].isna().all()
    shuffled = add_prior_day_history(example.iloc[::-1]).set_index('Inspection ID').sort_index()
    pd.testing.assert_frame_equal(checked.sort_index(), shuffled)

    training_history = add_prior_day_history(train_records)
    known = training_history.loc[training_history.establishment_key.notna()]
    assert known.prior_fails.le(known.prior_labelled).all()
    assert known.prior_labelled.le(known.prior_visits).all()
    assert known.loc[known.previous_visit_date.notna(), 'previous_visit_date'].lt(
        known.loc[known.previous_visit_date.notna(), 'Inspection Date']).all()
    assert training_history.loc[training_history.identity_unknown, 'prior_visits'].isna().all()
    assert len(training_history) == len(train_records)
    hist_eda = training_history.loc[training_history.fail.notna() & ~training_history.identity_unknown].copy()
    hist_eda['fail'] = hist_eda['fail'].astype(int)
    history_omitted = int((training_history.fail.notna() & training_history.identity_unknown).sum())
    print(f'Labelled training visits omitted only from history comparisons because identity is unknown: {history_omitted:,}')

    hist_eda['prior_group'] = pd.cut(hist_eda.prior_visits, bins=[-1, 0, 1, 2, 5, 10, 20, np.inf],
                                     labels=['0 observed', '1', '2', '3-5', '6-10', '11-20', '21+'])
    hist_eda['gap_group'] = pd.cut(hist_eda.days_since_previous, bins=[0, 90, 180, 365, 730, np.inf],
                                   labels=['1-90 days', '91-180', '181-365', '366-730', '>730'])
    hist_eda['gap_group'] = hist_eda.gap_group.cat.add_categories(['No prior observed']).fillna('No prior observed')
    prior_stats = rate_table(hist_eda, 'prior_group')
    gap_stats = rate_table(hist_eda, 'gap_group').reindex(['No prior observed', '1-90 days', '91-180', '181-365', '366-730', '>730'])
    display(prior_stats.round(4))
    display(gap_stats.round(4))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    draw_rate_bars(axes[0], prior_stats, 'Figure 7a. Prior observed visit count', sort_by_rate=False)
    draw_rate_bars(axes[1], gap_stats, 'Figure 7b. Days since prior observed visit', GREEN, sort_by_rate=False)
    for ax in axes:
        ax.set_xlim(0, 55)
    fig.tight_layout()
    plt.show()
""")

markdown("""
    **Finding:** history groups show different observed rates after identity and
    same-day corrections; read the recomputed values above rather than the old
    drafts' numbers. **Implication:** prior counts, prior labelled failure rate,
    recency and an explicit history-availability flag are candidate features.
    **Limits:** intervals and inspection type are related; repeated visits are not
    independent businesses; valid licences can still change or be reused. The
    construction cannot recover pre-2016 visits or establish causal effects.

    ### Figure 8. Does the Risk pattern persist within inspection types?

    The earlier drafts noticed higher aggregate failure rates in the Low group.
    Instead of assuming a cause, compare the same four inspection types within each
    named risk grade. Show both rates and sample sizes; suppress cells with n < 100.
    Also standardise the rates to a common inspection-type mix in this restricted
    subset. This checks one compositional explanation, not all confounding.
""")

code("""
    focus_types = ['canvass', 'complaint', 'license', 'canvass re-inspection']
    focus = eda.loc[eda.risk.isin(risk_order) & eda.inspection_type.isin(focus_types)]
    risk_type = focus.groupby(['risk', 'inspection_type'], observed=True).fail.agg(n='size', fail_count='sum', fail_rate='mean')
    rates = risk_type.fail_rate.unstack().reindex(index=risk_order, columns=focus_types)
    counts = risk_type.n.unstack().reindex(index=risk_order, columns=focus_types)
    supported = counts.ge(100) & rates.notna()
    display(risk_type.round(4))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    plotted = rates.where(supported).to_numpy(dtype=float) * 100
    mesh = ax.imshow(np.ma.masked_invalid(plotted), cmap='YlGnBu', vmin=0, vmax=60, aspect='auto')
    for i in range(len(risk_order)):
        for j in range(len(focus_types)):
            text = f'{plotted[i,j]:.1f}%\\nn={int(counts.iloc[i,j]):,}' if supported.iloc[i,j] else 'n < 100 / missing'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if supported.iloc[i,j] and plotted[i,j] > 35 else 'black')
    ax.set_xticks(range(len(focus_types)), ['Canvass', 'Complaint', 'License', 'Canvass re-inspection'], rotation=15, ha='right')
    ax.set_yticks(range(len(risk_order)), risk_order)
    ax.set_title('Figure 8. Risk differences within inspection type (training period)')
    fig.colorbar(mesh, ax=ax, label='Fail among labelled visits (%)')
    fig.tight_layout()
    plt.show()

    common_types = supported.columns[supported.all(axis=0)]
    assert len(common_types) > 0
    common_focus = focus.loc[focus.inspection_type.isin(common_types)]
    weights = common_focus.inspection_type.value_counts(normalize=True).reindex(common_types)
    assert np.isclose(weights.sum(), 1)
    standardised = rates[common_types].mul(weights, axis='columns').sum(axis=1)
    crude_restricted = common_focus.groupby('risk').fail.agg(n='size', fail_rate='mean').reindex(risk_order)
    standardisation = crude_restricted.rename(columns={'fail_rate': 'crude_rate_in_same_subset'})
    standardisation['common_mix_rate'] = standardised
    standardisation['percentage_point_change'] = 100 * (standardisation.common_mix_rate - standardisation.crude_rate_in_same_subset)
    display(standardisation.round(4))
    print('Common inspection-type weights:', weights.round(4).to_dict())
    print('Low minus High after common-mix standardisation (percentage points):',
          round(100 * (standardised.loc['Risk 3 (Low)'] - standardised.loc['Risk 1 (High)']), 2))
""")

markdown("""
    **Finding:** the Low-versus-High difference is present within several inspection
    types; a common type mix changes the aggregate comparison but does not by itself
    explain away the pattern in the supported subset. **Implication:** retain Risk
    as a candidate category and investigate type/facility interactions rather than
    imposing a monotonic probability interpretation. **Limits:** this is descriptive
    standardisation over four types, not a causal adjustment or a model performance
    test. Facility mix, timing and inspection selection remain possible explanations.
""")

markdown("""
    ## 4. Evidence-based data-quality summary

    Counts below are computed from the saved snapshot unless explicitly marked
    training-period. “Possible reason” is a hypothesis. Repeated visits are valid
    observations; an identity or timing problem changes linkage, not automatically
    the eligibility of a supervised label.
""")

code("""
    positive_visits = df.loc[~df.identity_unknown, 'establishment_key'].value_counts()
    same_days = df.loc[~df.identity_unknown].groupby(['establishment_key', 'Inspection Date']).size()
    same_day_groups = int(same_days.gt(1).sum())
    same_day_rows = int(same_days.loc[same_days.gt(1)].sum())
    quality_rows = [
        ['Rounded missing rates', f'Licence NA: {missing_licence.sum():,} / {len(df):,} ({100 * missing_licence.mean():.4f}%); displays as 0.00% at two decimals', 'Display rounding', 'Report count and precise percentage; zero displayed percentage does not prove completeness.'],
        ['Zero / invalid identity', f'Zero licences: {zero_licence.sum():,}; missing: {missing_licence.sum():,}; other invalid: {other_invalid.sum():,}', 'Unknown source coding or incomplete entry', 'Retain visits; no cross-row linkage for invalid identity; add unknown-history flag.'],
        ['Zero identity collisions', f'Zero rows contain {raw.loc[zero_licence, "DBA Name"].nunique():,} names and {raw.loc[zero_licence, "Address"].nunique():,} addresses', 'Zero does not identify one establishment', 'Never aggregate all zero licences into one history.'],
        ['Repeated legitimate visits', f'{positive_visits.size:,} positive licence strings; maximum {positive_visits.max():,} visits', 'Repeat inspection process; imperfect licence proxy', 'Keep distinct visits and describe known/unseen establishment evaluation.'],
        ['Same-day ambiguity', f'{same_day_groups:,} valid licence-days contain multiple visits ({same_day_rows:,} rows)', 'Multiple visits or records on one date; timing unverified', 'Build strictly prior-day history, not ID-ordered within-day history.'],
        ['Unlabelled outcomes', f'{df.fail.isna().sum():,} / {len(df):,} rows have no project label', 'Operational outcomes without a project judgement label', 'Exclude from supervised labels; count eligible earlier visits separately.'],
        ['Post-start information', 'Current Results and Violations describe this inspection; Figure 3 shows outcome-associated missingness', 'Recorded during/after the visit', 'Target only for Results; exclude current Violations, missing flag and derived counts.'],
        ['Facility / inspection text', f'Raw nonmissing categories: facility {raw["Facility Type"].nunique():,}, inspection {raw["Inspection Type"].nunique():,}', 'Case, spacing, free text and substantive variants', 'Use fixed normalisation; learn rare-level handling in training only; verify semantic merges.'],
        ['Missing categories', f'Facility NA {raw["Facility Type"].isna().sum():,}; Risk NA {raw.Risk.isna().sum():,}; Inspection Type NA {raw["Inspection Type"].isna().sum():,}', 'Incomplete recording, mechanism unconfirmed', 'Keep Unknown categories; do not infer low risk or remove all such visits.'],
        ['City / state heterogeneity', f'{raw.City.nunique():,} nonmissing City values; {raw.State.nunique():,} State values', 'Verified aliases plus genuinely different entries', 'Do not map every value to Chicago/IL; validate aliases before geographic cleaning.'],
        ['Geographic gaps / redundancy', f'Latitude missing {raw.Latitude.isna().sum():,}; coordinate mismatches {coordinate_mismatches:,}', 'Missing geocoding or entry; true cause unconfirmed', 'Location redundant with coordinates here; ZIP is postal area; preserve missingness if coordinates used.'],
        ['Snapshot coverage', f'Actual last date {df["Inspection Date"].max().date()}; final month {monthly.iloc[-1]:,} visits', 'Partial calendar coverage and possible later additions', 'Label partial periods; pin snapshot and do not assert a fixed publication lag.'],
        ['Procedure comparability', 'Official portal flags an inspection-procedure change on 2018-07-01', 'Documented change to the inspection process; effect size not established here', 'Consider temporal sensitivity checks in A2; do not attribute an observed rate change to the policy without evidence.'],
        ['Duplicate / date audit', f'Exact duplicates {exact_duplicates:,}; duplicate IDs {duplicate_ids:,}; invalid dates {invalid_date_count:,}', 'No evidence of these failures in this snapshot', 'Keep all rows; fail clearly if a later version violates assumptions.'],
    ]
    quality = pd.DataFrame(quality_rows, columns=['problem', 'evidence', 'possible reason', 'effect on modelling'])
    with pd.option_context('display.max_colwidth', None):
        display(quality)
""")

markdown("""
    ## 5. Feature availability and evaluation population

    A field can be allowed before inspection yet unnecessary for the first model.
    Excluding a field from model input does not mean deleting it from raw data.
    Identity-unknown visits remain eligible for prediction using other fields.
""")

code("""
    availability = pd.DataFrame([
        ['Inspection ID', 'Identifier', 'Preserve for matching predictions; no numeric model input.'],
        ['DBA Name, AKA Name', 'Pre-visit candidates', 'Keep for audit; omit raw names from first baseline.'],
        ['License #', 'Identity with caveats', 'Link only valid positive keys; identity_unknown for the rest; no raw-ID feature.'],
        ['Facility Type, Risk, Inspection Type', 'Pre-visit candidates', 'Normalise; encode categorically; fit category handling on training only.'],
        ['Address, City, State', 'Pre-visit candidates', 'Keep for audit/linkage checks; omit raw text in first baseline.'],
        ['Zip', 'Pre-visit candidate', 'Categorical postal area; not community or ordinal magnitude.'],
        ['Latitude, Longitude', 'Optional pre-visit candidates', 'Compare later against ZIP; handle missingness and validate outliers.'],
        ['Location', 'Redundant representation', 'Not an additional feature once coordinates are validated.'],
        ['Inspection Date', 'Calendar / evaluation', 'Chronological splits; month and year features may be derived.'],
        ['Results', 'Target only for current visit', 'Map allowed outcomes; only eligible historical outcomes may enter history.'],
        ['Violations', 'Unavailable before current visit', 'Exclude current content, missingness and all its derivatives.'],
        ['Prior counts, rates, recency', 'Derived historical candidates', 'Strictly earlier available records; no hidden-period outcomes.'],
    ], columns=['field(s)', 'role', 'decision / rule'])
    with pd.option_context('display.max_colwidth', None):
        display(availability)

    fit_keys = set(eda.establishment_key.dropna())
    historical_keys = set(train_records.establishment_key.dropna())
    split_windows = [
        ('train', df['Inspection Date'] < VALID_START),
        ('validation', (df['Inspection Date'] >= VALID_START) & (df['Inspection Date'] < TEST_START)),
        ('internal evaluation', (df['Inspection Date'] >= TEST_START) & (df['Inspection Date'] <= INTERNAL_END)),
    ]
    split_rows = []
    for name, mask in split_windows:
        part = df.loc[mask & df.fail.notna()]
        valid_identity = ~part.identity_unknown
        split_rows.append({
            'split': name,
            'actual_dates': f'{part["Inspection Date"].min().date()} to {part["Inspection Date"].max().date()}',
            'labelled_rows': len(part),
            'unknown_identity_rows': int(part.identity_unknown.sum()),
            'seen_in_labelled_train_percent_of_all_rows': 100 * (valid_identity & part.establishment_key.isin(fit_keys)).mean(),
            'known_in_any_training_history_percent_of_all_rows': 100 * (valid_identity & part.establishment_key.isin(historical_keys)).mean(),
            'unseen_in_training_history_rows': int((valid_identity & ~part.establishment_key.isin(historical_keys)).sum()),
        })
    split_summary = pd.DataFrame(split_rows)
    display(split_summary.round(2))
    print('No validation/internal-evaluation failure rates or model scores are calculated here.')
""")

markdown("""
    ## 6. Modelling and evaluation plan (approximately one page)

    **Task and baselines.** Predict Fail = 1 versus Pass / Pass w/ Conditions = 0.
    Compare an always-non-Fail reference with Logistic Regression. Start with
    normalised Facility Type, Risk, Inspection Type, categorical ZIP and calendar
    month/year. Compare adding prior-day counts, prior labelled failure rate,
    recency and explicit identity/history-availability flags. Unknown identities
    retain other predictors. Raw names, IDs, address text, current Results and
    current Violations (including missingness and derived values) are excluded.
    Coordinates are an optional later comparison; Location is redundant.

    **Preparation and history.** Fit encoders, imputers, rare-level rules and any
    scaling on training only, with safe unseen-category handling. Treat Risk as
    categorical. Distinguish a valid key with no observed history, a valid key with
    no prior labelled visits, and unknown identity. Historical failure rates divide
    by prior labelled counts; do not fill undefined rates with an assertion of zero
    failure. Compare explicit missingness with smoothing toward a training-only
    base rate; fit smoothing strength inside chronological development folds.
    For a training row, any smoothing base rate must also come from an earlier
    development window. Only strictly earlier-day records may contribute history.

    **Time and information boundaries.** Use 2016-2024 for training, 2025 for
    validation and 2026-01 through 2026-09 for internal model evaluation. Fix these
    windows before model experiments. Earlier drafts explored later-period outcomes,
    so the internal period is not claimed to be a wholly untouched test; disclose
    that limitation. Do not further tune based on its results. Inside training,
    use expanding time windows, with same-date records kept on the same side of
    each boundary. Compare model settings/class weights there; select the operating
    threshold on validation, beginning with 0.5. Freeze all choices before internal
    evaluation. For each scored validation/evaluation block, freeze outcome history
    at the block start rather than using earlier true labels from that scored block.
    This is conservative and matches batch prediction when block outcomes are not
    available. The official hidden window is separate; its outcomes are never used.

    **Groups, metrics and limits.** Report Macro-F1 using the specified two-label
    calculation, plus per-class precision/recall/F1 and a confusion matrix. Also
    report known-history, unseen-valid-licence and unknown-identity groups when
    sample sizes permit. Licence overlap over time is not itself leakage; it tells
    us which deployment population is being evaluated. A missed Fail is an unflagged
    failed inspection; a false alarm flags an eventual non-Fail. Macro-F1 weights
    class F1 values equally, not operational costs. Associations concern inspected,
    labelled visits, not all businesses or causal effects. Nonstationarity, partial
    publication coverage, the 2018 procedure change, imperfect identity and absent pre-2016 history limit
    interpretation. No trained model or claimed performance is part of this A1 EDA.
""")

markdown("""
    ## 7. Conclusions and submission checks

    - Correct labels and denominators matter: unlabelled operational visits are
      neither Fail nor non-Fail examples, but may describe eligible past history.
    - A displayed `0.00%` missing rate and a literal zero licence are different
      issues. Counts reveal the first; identity auditing reveals the second.
    - Time, inspection type, facilities and postal areas show useful descriptive
      variation. Risk differences persist within several supported inspection types;
      the common-mix check narrows one explanation without claiming a cause.
    - Historical comparisons must exclude same-day outcomes and invalid-key linkage.
      Their candidate features need later time-respecting model validation.
    - Current-inspection Results/Violations and their post-start derivatives cannot
      be prediction inputs. Preserve the saved snapshot and audit trail.

    **Before submission:** team members should review the revised choices and be
    able to explain the work; fill in the actual team names/IDs as required by the
    course platform; confirm the notebook and saved inputs are submitted/linked in
    the required repository structure. A later pre-cutoff refresh is optional, not
    a prerequisite; it requires explicit version changes and a full rerun. This
    notebook is an executed A1 draft, not a claim of guaranteed marks. Model fitting,
    official predictions and the commissioner brief belong to Assessment 2.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata['language_info'] = {'name': 'python'}
nbf.validate(notebook)
output = ROOT / 'base' / 'EDA_food_inspections.ipynb'
nbf.write(notebook, output)
print(f'Built {output.name}: {len(cells)} cells, {sum(c.cell_type == "code" for c in cells)} code cells')
