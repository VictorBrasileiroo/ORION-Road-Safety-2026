"""EXP-17 reusable deterministic matching and conservative interval aggregation."""
import json
import numpy as np
import pandas as pd

YEARS = [2024, 2025]
TOL = 0.001  # kilometres; cumulative materiality threshold, not per-cell forgiveness
REGIONS = {}
for region, states in {
    'Norte': 'AC AP AM PA RO RR TO', 'Nordeste': 'AL BA CE MA PB PE PI RN SE',
    'Centro-Oeste': 'DF GO MT MS', 'Sudeste': 'ES MG RJ SP', 'Sul': 'PR RS SC',
}.items():
    REGIONS.update(dict.fromkeys(states.split(), region))
COUNTS = {'n_accidents': None, 'n_critical': 'critico', 'n_fatal': 'fatal',
          'n_serious': 'grave', 'n_no_victims': 'sem_vitimas'}

def normalize_br(series):
    values = pd.to_numeric(series, errors='coerce')
    valid = np.isfinite(values) & values.between(1, 999) & (values == np.floor(values))
    return values.where(valid).astype('Int64')

def match_accidents(prf, dim):
    prf = prf.reset_index(drop=True)
    bridge = pd.DataFrame({'accident_key': prf.accident_key, 'year': prf.year,
                           'uf': prf.uf_limpa, 'br_raw': prf.br.astype('string'),
                           'br_normalized': normalize_br(prf.br),
                           'km': pd.to_numeric(prf.km_num, errors='coerce')})
    bridge['br_text'] = bridge.br_normalized.astype('string').str.zfill(3)
    bridge['segment_id'] = pd.Series(pd.NA, index=bridge.index, dtype='string')
    bridge['segment_match_count'] = 0
    bridge['segment_match_status'] = 'UNMATCHED'
    bridge['segment_match_method'] = 'UF_BR_KM_INTERVAL'
    bridge['match_quality'] = 'UNRESOLVED'
    bridge['match_notes'] = ''
    bridge['candidate_segment_ids'] = '[]'
    bridge['unmatched_reason'] = ''
    # Mutable numpy arrays avoid quadratic Arrow string reallocations per cell.
    assigned = np.full(len(bridge), None, dtype=object)
    candidate_json = np.full(len(bridge), '[]', dtype=object)
    status = bridge.segment_match_status.to_numpy(dtype=object, copy=True)
    notes = np.full(len(bridge), '', dtype=object)
    reasons = np.full(len(bridge), '', dtype=object)
    invalid_br = bridge.br_normalized.isna()
    invalid_loc = ~bridge.uf.isin(REGIONS) | ~np.isfinite(bridge.km) | (bridge.km < 0)
    bridge.loc[invalid_br, ['segment_match_status', 'unmatched_reason']] = 'INVALID_BR'
    bridge.loc[~invalid_br & invalid_loc, 'segment_match_status'] = 'INVALID_LOCATION'
    bridge.loc[~invalid_br & invalid_loc, 'unmatched_reason'] = 'INVALID_LOCATION'
    status[:] = bridge.segment_match_status.to_numpy(dtype=object)
    reasons[:] = bridge.unmatched_reason.to_numpy(dtype=object)
    groups = {(uf, int(br)): g for (uf, br), g in dim.groupby(['uf', 'br'])}
    for key, accidents in bridge.loc[~invalid_br & ~invalid_loc].groupby(['uf', 'br_normalized']):
        segments = groups.get((key[0], int(key[1])))
        idx = accidents.index
        if segments is None:
            reasons[idx] = 'NO_SNV_BR_UF'
            continue
        starts, ends = segments.km_start.to_numpy(), segments.km_end.to_numpy()
        ids = segments.segment_id.to_numpy()
        terminal = ends.max()
        # Local UF+BR matrix only; chunk bounds avoid a global Cartesian product.
        for begin in range(0, len(accidents), 2048):
            block = accidents.iloc[begin:begin + 2048]
            km = block.km.to_numpy()[:, None]
            matches = (km >= starts) & ((km < ends) | ((km == ends) & (ends == terminal)))
            counts = matches.sum(axis=1)
            bridge.loc[block.index, 'segment_match_count'] = counts
            for row, mask, count in zip(block.index, matches, counts):
                candidates = sorted(ids[mask].tolist())
                candidate_json[row] = json.dumps(candidates)
                if count == 1:
                    assigned[row] = candidates[0]
                    status[row] = 'MATCHED_UNIQUE'
                elif count > 1:
                    status[row] = 'AMBIGUOUS'
                    notes[row] = 'Overlapping canonical intervals; no final assignment'
                else:
                    value = bridge.at[row, 'km']
                    reason = 'KM_OUTSIDE_SNV_RANGE' if value < starts.min() or value > terminal else 'OTHER'
                    if np.any(np.abs(ends - value) <= TOL) or np.any(np.abs(starts - value) <= TOL):
                        reason = 'BOUNDARY_CASE'
                    reasons[row] = reason
    bridge['segment_id'] = pd.array(assigned, dtype='string')
    bridge['candidate_segment_ids'] = candidate_json
    bridge['segment_match_status'] = status
    bridge['match_notes'] = notes
    bridge['unmatched_reason'] = reasons
    bridge['match_quality'] = np.where(status == 'MATCHED_UNIQUE', 'DETERMINISTIC', 'UNRESOLVED')
    return bridge

def aggregate_partition(segment, year, group):
    start, end = float(segment.km_start), float(segment.km_end)
    expected = end - start
    result = dict(segment_id=segment.segment_id, snv_code=segment.snv_code, year=year,
                  vmda=np.nan, vmda_partial=np.nan, pnct_expected_km=expected,
                  pnct_covered_km=0., pnct_vmda_observed_km=0., pnct_coverage_ratio=0.,
                  pnct_n_subsegments=0, pnct_n_vmda_observed=0,
                  pnct_has_overlap=False, pnct_has_gap=False, pnct_has_missing_vmda=False,
                  pnct_has_other_quality_issue=False, pnct_gap_km=0., pnct_overlap_km=0.,
                  pnct_missing_vmda_km=0., pnct_outside_km=0., pnct_observed_coverage_ratio=0.,
                  pnct_status='NO_PNCT', vmda_reference_year=year,
                  vmda_window_start='2024-07-01' if year == 2025 else None,
                  vmda_window_end='2025-06-30' if year == 2025 else None,
                  pnct_network_versions='[]', pnct_classifications='[]',
                  pnct_has_referential_estimate=False)
    if group is None or group.empty:
        return result, []
    group = group.sort_values(['km_start_best', 'km_end_best', 'source_segment_key'])
    result['pnct_n_subsegments'] = len(group)
    volume = pd.to_numeric(group.vmda_total, errors='coerce').to_numpy(dtype=float)
    valid_volume = np.isfinite(volume) & (volume > 0)
    result['pnct_n_vmda_observed'] = int(valid_volume.sum())
    result['pnct_has_missing_vmda'] = bool((~valid_volume).any())
    result['pnct_network_versions'] = json.dumps(sorted(group.network_version.dropna().unique().tolist()))
    result['pnct_classifications'] = json.dumps(sorted(group.classification_canonical.dropna().unique().tolist()))
    result['pnct_has_referential_estimate'] = bool(group.is_referential_explicit.fillna(False).any())
    lo = group.km_start_best.to_numpy(dtype=float)
    hi = group.km_end_best.to_numpy(dtype=float)
    ext = group.extension_best.to_numpy(dtype=float)
    valid_interval = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    original_ok = (np.abs(group.km_start_original.to_numpy() - start) <= TOL) & (np.abs(group.km_end_original.to_numpy() - end) <= TOL)
    identity_ok = (group.uf.to_numpy() == segment.uf) & (normalize_br(group.br).to_numpy() == segment.br)
    extension_ok = np.isfinite(ext) & (ext > 0) & (np.abs(ext - (hi - lo)) <= TOL + 1e-10)
    version_ok = group.network_version.eq('202401A').to_numpy()
    quality = valid_interval & original_ok & identity_ok & extension_ok & version_ok
    result['pnct_has_other_quality_issue'] = bool((~quality).any())
    clipped_lo, clipped_hi = np.maximum(lo, start), np.minimum(hi, end)
    outside = np.where(valid_interval, np.maximum(0., start - lo) + np.maximum(0., hi - end), 0.)
    result['pnct_outside_km'] = float(outside.sum())
    if outside.sum() > TOL + 1e-10:
        result['pnct_has_other_quality_issue'] = True
    cuts = np.unique(np.concatenate(([start, end], clipped_lo[valid_interval], clipped_hi[valid_interval])))
    cuts = cuts[(cuts >= start) & (cuts <= end)]
    covered = observed = numerator = gaps = overlaps = missing = 0.
    contributions = np.zeros(len(group))
    # Elementary interval sweep counts union coverage and avoids overlap double counting.
    for left, right in zip(cuts[:-1], cuts[1:]):
        width = right - left
        middle = (left + right) / 2
        active = np.flatnonzero(valid_interval & (lo <= middle) & (hi > middle))
        if len(active) == 0:
            gaps += width
        else:
            covered += width
            if len(active) > 1:
                overlaps += width
            elif valid_volume[active[0]]:
                i = active[0]
                observed += width
                numerator += volume[i] * width
                contributions[i] += width
            else:
                missing += width
    result.update(pnct_covered_km=covered, pnct_vmda_observed_km=observed,
                  pnct_coverage_ratio=covered / expected,
                  pnct_observed_coverage_ratio=observed / expected,
                  pnct_gap_km=gaps, pnct_overlap_km=overlaps, pnct_missing_vmda_km=missing,
                  pnct_has_gap=gaps > TOL + 1e-10,
                  pnct_has_overlap=overlaps > TOL + 1e-10,
                  vmda_partial=numerator / observed if observed > 0 else np.nan)
    if result['pnct_has_other_quality_issue']:
        status = 'OTHER_QUALITY_ISSUE'
    elif result['pnct_has_overlap']:
        status = 'OVERLAP'
    elif result['pnct_has_gap']:
        status = 'GAP'
    elif observed <= 0 or expected - observed > TOL + 1e-10:
        status = 'MISSING_VMDA' if observed == 0 else 'PARTIAL'
    else:
        status = 'VALID'
        result['vmda'] = result['vmda_partial']
    result['pnct_status'] = status
    records = []
    for i, (_, row) in enumerate(group.iterrows()):
        record = row.to_dict()
        record.update(segment_id=segment.segment_id,
                      canonical_km_start=start, canonical_km_end=end,
                      clipped_length_km=max(0., clipped_hi[i] - clipped_lo[i]) if valid_interval[i] else 0.,
                      covered_length_i=contributions[i],
                      interval_quality_ok=bool(quality[i]))
        records.append(record)
    return result, records

def aggregate_pnct(dim, pnct):
    groups = dict(tuple(pnct.groupby(['snv_code', 'year'])))
    aggregates, records = [], []
    for segment in dim.itertuples():
        for year in YEARS:
            result, subsegments = aggregate_partition(segment, year, groups.get((segment.snv_code, year)))
            aggregates.append(result)
            records.extend(subsegments)
    return pd.DataFrame(aggregates), pd.DataFrame(records)

def segment_fact(dim, accidents, pnct):
    fields = ['segment_id', 'snv_code', 'snv_version', 'uf', 'br', 'region', 'km_start', 'km_end', 'length_km']
    skeleton = dim[fields].merge(pd.DataFrame({'year': YEARS}), how='cross')
    agg = accidents.groupby(['segment_id', 'year']).agg(
        n_accidents=('accident_key', 'size'), n_critical=('critico', 'sum'),
        n_fatal=('fatal', 'sum'), n_serious=('grave', 'sum'), n_no_victims=('sem_vitimas', 'sum')).reset_index()
    result = skeleton.merge(agg, on=['segment_id', 'year'], how='left', validate='one_to_one')
    result[list(COUNTS)] = result[list(COUNTS)].fillna(0).astype('int64')
    result = result.merge(pnct, on=['segment_id', 'snv_code', 'year'], how='left', validate='one_to_one')
    valid = result.vmda.notna() & (result.vmda > 0) & (result.length_km > 0) & result.pnct_status.eq('VALID')
    result['annual_vehicle_km_proxy'] = (result.vmda * 365 * result.length_km).where(valid)
    for prefix, count in [('accidents', 'n_accidents'), ('critical', 'n_critical'), ('fatal', 'n_fatal'), ('serious', 'n_serious')]:
        result[f'{prefix}_per_100m_vehicle_km'] = result[count] / result.annual_vehicle_km_proxy * 100000000
    result['rate_semantics'] = 'approximate_exposure_rate'
    return result

def territorial_mart(fact, key):
    rows = []
    for (territory, year), group in fact.groupby([key, 'year']):
        observed = group[group.annual_vehicle_km_proxy.notna() & (group.annual_vehicle_km_proxy > 0)]
        exposure = observed.annual_vehicle_km_proxy.sum(min_count=1)
        row = {key: territory, 'year': year, 'n_segment_years': len(group),
               'n_segment_years_valid_exposure': len(observed),
               'annual_vehicle_km_proxy': exposure,
               'length_km_total': group.length_km.sum(),
               'length_km_valid_exposure': observed.length_km.sum(),
               'rate_semantics': 'approximate_exposure_rate',
               'rate_population': 'segments_with_valid_exposure_only'}
        for count in COUNTS:
            row[count] = group[count].sum()
            row[count + '_valid_exposure'] = observed[count].sum()
        for prefix, count in [('accidents', 'n_accidents'), ('critical', 'n_critical'), ('fatal', 'n_fatal'), ('serious', 'n_serious')]:
            row[prefix + '_per_100m_vehicle_km'] = observed[count].sum() / exposure * 100000000 if pd.notna(exposure) and exposure > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
