import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import OneHotEncoder
import os
from typing import Tuple, List, Dict, Optional, Any
import logging

# Optional: Import tqdm for progress bar
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

def inv_signed_log(y: float) -> float:
    """Inverse of signed_log transformation: sign(y) * (exp(abs(y)) - 1)."""
    if pd.isna(y):
        return np.nan
    try:
        abs_y = np.abs(np.float64(y))
        exp_val = np.exp(abs_y)
        if np.isinf(exp_val):
            return np.inf * np.sign(y)
        return np.sign(y) * (exp_val - 1)
    except OverflowError:
        return np.inf * np.sign(y)

def robust_slope(series: pd.Series, periods: list) -> Tuple[float, float]:
    """
    Calculate slope and R-squared robustly using linear regression.
    Handles NaN, insufficient data (< 2 points), and constant data explicitly.
    Uses the series index (expected to be period numbers) as the independent variable.
    Returns: tuple[float, float]: (slope, r_squared). Returns (0.0, 0.0) for insufficient/constant data or errors.
    """
    if not periods or series.empty:
        return 0.0, 0.0

    valid_indices = series.index.intersection(periods)
    if valid_indices.empty:
        return 0.0, 0.0
    data = series.loc[valid_indices].dropna()

    if len(data) < 2:
        return 0.0, 0.0
    if data.nunique() == 1:
        return 0.0, 0.0  # Slope is 0, R^2 is ill-defined (treat as 0 fit)

    x_values = data.index.astype(float)
    y_values = data.values
    try:
        slope, _, r_value, p_value, std_err = linregress(x_values, y_values)
        if pd.isna(slope) or pd.isna(r_value):
            return 0.0, 0.0
        r_squared = r_value**2
        return slope, r_squared
    except ValueError as e:
        print(f"Warning: linregress failed unexpectedly for index {data.index.tolist()}: {e}. Returning (0.0, 0.0).")
        return 0.0, 0.0

def calculate_stdev(series: pd.Series, periods: list) -> float:
    """
    Calculate sample standard deviation robustly for specified periods.
    Ignores NaNs. Requires at least 2 data points. Uses ddof=1. Returns 0.0 for constant data.
    Returns: float: Standard deviation, or np.nan if fewer than 2 data points.
    """
    if not periods or series.empty:
        return np.nan

    valid_indices = series.index.intersection(periods)
    if valid_indices.empty:
        return np.nan
    data = series.loc[valid_indices].dropna()

    if len(data) < 2:
        return np.nan
    if data.nunique() == 1:
        return 0.0  # Standard deviation of constant data is 0
    return np.std(data.values, ddof=1)

def process_group(
    group: pd.DataFrame,
    period_range: List[int],
    raw_scaled_sales_signed_log_cols: List[str],
    ratio_signed_log_cols: List[str],
    cstat_std_cols: List[str],
    target_metric: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process a single group of data to generate features.
    """
    # Debug logging for input
    logger.info(f"Processing group with columns: {group.columns.tolist()}")
    logger.info(f"Ratio columns in group: {[col for col in group.columns if '_RATIO_SIGNED_LOG' in col]}")
    
    feats = {'ID': group['ID'].iloc[0], 'PIT_DATE': group['PIT_DATE'].iloc[0]}

    # --- Handle duplicate PERIODs ---
    if group['PERIOD'].duplicated().any():
        n_dups = group['PERIOD'].duplicated().sum()
        print(f"Warning: Found {n_dups} duplicate PERIOD(s) in group ID {group['ID'].iloc[0]}, PIT {group['PIT_DATE'].iloc[0]}. Keeping first.")
        group = group.drop_duplicates(subset=['PERIOD'], keep='first').copy()

    # --- Set Index ---
    try:
        group = group.set_index('PERIOD')
        if not group.index.is_unique:
            raise ValueError("Index not unique after dropping duplicates - unexpected.")
        if not group.index.is_monotonic_increasing:
            group = group.sort_index()
    except Exception as e:
        print(f"Error setting index for group ID {group['ID'].iloc[0]}, PIT {group['PIT_DATE'].iloc[0]}: {e}. Skipping.")
        return {}

    # Define index covering full range and period subsets
    full_range_index = pd.Index(period_range, name='PERIOD')  # Includes 0
    hist_periods = [p for p in period_range if p <= 0]
    fwd_periods = [p for p in period_range if p > 0]  # Include all forward periods
    combined_periods = period_range  # Include all periods

    numerical_feature_keys = []

    # --- Process Metric Groups ---
    for metric_group, metric_cols in [('scaled', raw_scaled_sales_signed_log_cols),
                                    ('ratio', ratio_signed_log_cols)]:
        for metric in metric_cols:
            if metric not in group.columns:
                logger.info(f"Metric {metric} not found in group columns")
                continue
                
            # Debug logging for each metric
            logger.info(f"Processing {metric_group} metric: {metric}")
            
            # For target metric, exclude PERIOD=1 from calculations
            if metric == target_metric:
                current_fwd_periods = [p for p in fwd_periods if p != 1]
                current_combined_periods = [p for p in combined_periods if p != 1]
                logger.info(f"Target metric {metric} - using forward periods: {current_fwd_periods}")
            else:
                current_fwd_periods = fwd_periods
                current_combined_periods = combined_periods
            
            series = group[metric].reindex(full_range_index)

            # Find specific period values
            negative_hist_periods = [p for p in hist_periods if p < 0]
            valid_negative_hist_indices = series.loc[series.index.intersection(negative_hist_periods)].dropna().index
            valid_fwd_indices = series.loc[series.index.intersection(current_fwd_periods)].dropna().index

            latest_negative_hist_period = valid_negative_hist_indices.max() if not valid_negative_hist_indices.empty else np.nan
            first_fwd_period = valid_fwd_indices.min() if not valid_fwd_indices.empty else np.nan
            if metric == target_metric:
                logger.info(f"Target metric {metric} - first forward period: {first_fwd_period}")
            value_period_0 = series.get(0, np.nan)

            # Feature: Levels
            level_latest_neg_hist_key = f'level_latest_neg_hist_{metric}'
            level_period_0_key = f'level_period_0_{metric}'
            level_first_fwd_key = f'level_first_fwd_{metric}'
            feats[level_latest_neg_hist_key] = series.get(latest_negative_hist_period, np.nan)
            feats[level_period_0_key] = value_period_0
            feats[level_first_fwd_key] = series.get(first_fwd_period, np.nan)
            numerical_feature_keys.extend([level_latest_neg_hist_key, level_period_0_key, level_first_fwd_key])

            # Features: Slopes and R-squared
            hist_slope, hist_r2 = robust_slope(series, hist_periods)
            fwd_slope, fwd_r2 = robust_slope(series, current_fwd_periods)
            combined_slope, combined_r2 = robust_slope(series, current_combined_periods)

            hist_slope_key = f'{metric_group}_hist_slope_{metric}'
            fwd_slope_key = f'{metric_group}_fwd_slope_{metric}'
            combined_slope_key = f'{metric_group}_combined_slope_{metric}'
            hist_r2_key = f'{metric_group}_hist_r2_{metric}'
            fwd_r2_key = f'{metric_group}_fwd_r2_{metric}'
            combined_r2_key = f'{metric_group}_combined_r2_{metric}'
            feats.update({
                hist_slope_key: hist_slope, fwd_slope_key: fwd_slope, combined_slope_key: combined_slope,
                hist_r2_key: hist_r2, fwd_r2_key: fwd_r2, combined_r2_key: combined_r2
            })
            numerical_feature_keys.extend([
                hist_slope_key, fwd_slope_key, combined_slope_key,
                hist_r2_key, fwd_r2_key, combined_r2_key
            ])

            # Features: Volatility
            hist_vol = calculate_stdev(series, hist_periods)
            fwd_vol = calculate_stdev(series, current_fwd_periods)
            combined_vol = calculate_stdev(series, current_combined_periods)

            hist_vol_key = f'{metric_group}_hist_vol_{metric}'
            fwd_vol_key = f'{metric_group}_fwd_vol_{metric}'
            combined_vol_key = f'{metric_group}_combined_vol_{metric}'
            feats[hist_vol_key] = hist_vol
            feats[fwd_vol_key] = fwd_vol
            feats[combined_vol_key] = combined_vol
            numerical_feature_keys.extend([hist_vol_key, fwd_vol_key, combined_vol_key])

            # Features: Normalized Slopes
            norm_hist_slope_key = f'{metric_group}_norm_hist_slope_{metric}'
            norm_fwd_slope_key = f'{metric_group}_norm_fwd_slope_{metric}'
            feats[norm_hist_slope_key] = hist_slope / hist_vol if pd.notna(hist_vol) and hist_vol != 0 else np.nan
            feats[norm_fwd_slope_key] = fwd_slope / fwd_vol if pd.notna(fwd_vol) and fwd_vol != 0 else np.nan
            numerical_feature_keys.extend([norm_hist_slope_key, norm_fwd_slope_key])

            # Feature: Slope Divergence
            slope_divergence_key = f'{metric_group}_slope_divergence_{metric}'
            feats[slope_divergence_key] = fwd_slope - hist_slope if pd.notna(fwd_slope) and pd.notna(hist_slope) else np.nan
            numerical_feature_keys.append(slope_divergence_key)

            # Feature: Acceleration (Slope of Differences)
            diff_series = series.diff()
            min_hist_period = min(hist_periods) if hist_periods else None
            valid_hist_diff_periods = [
                p for p in hist_periods
                if p in diff_series.index and not pd.isna(diff_series.get(p)) and (min_hist_period is None or p > min_hist_period)
            ]
            valid_fwd_diff_periods = [p for p in current_fwd_periods if p in diff_series.index and not pd.isna(diff_series.get(p))]

            hist_accel, _ = robust_slope(diff_series, valid_hist_diff_periods)
            fwd_accel, _ = robust_slope(diff_series, valid_fwd_diff_periods)
            hist_accel_key = f'{metric_group}_hist_accel_{metric}'
            fwd_accel_key = f'{metric_group}_fwd_accel_{metric}'
            feats[hist_accel_key] = hist_accel
            feats[fwd_accel_key] = fwd_accel
            numerical_feature_keys.extend([hist_accel_key, fwd_accel_key])

    # --- Process Relative Dispersion ---
    for std_col in cstat_std_cols:
        estimate_col = std_col.replace('_CSTAT_STD', '')
        if not all(col in group.columns for col in [estimate_col, std_col]):
            continue

        std_series = group[std_col]
        estimate_series = group[estimate_col]

        for fwd_period in fwd_periods:
            rel_disp_key = f'rel_disp_{std_col}_period_{fwd_period}'
            numerical_feature_keys.append(rel_disp_key)
            try:
                slog_estimate = estimate_series.get(fwd_period)
                slog_std = std_series.get(fwd_period)

                if pd.isna(slog_estimate) or pd.isna(slog_std):
                    feats[rel_disp_key] = np.nan
                    continue

                actual_estimate = inv_signed_log(slog_estimate)
                actual_stdev = inv_signed_log(slog_std)

                if pd.isna(actual_stdev) or actual_stdev < 0 or not np.isfinite(actual_stdev):
                    feats[rel_disp_key] = np.nan
                    continue
                if pd.isna(actual_estimate) or not np.isfinite(actual_estimate):
                    feats[rel_disp_key] = np.nan
                    continue

                denominator = max(abs(actual_estimate), 1e-9)
                if denominator == 0:
                    feats[rel_disp_key] = np.nan
                    continue
                relative_dispersion_log1p = np.log1p(actual_stdev / denominator)
                feats[rel_disp_key] = relative_dispersion_log1p
            except Exception as e:
                feats[rel_disp_key] = np.nan
                print('Error: Relative dispersion calc failed unexpectedly for ID {}, key {}: {}'.format(
                    group['ID'].iloc[0], rel_disp_key, e))

    # --- Process As-is Ratio Values ---
    for period in period_range:
        for ratio_metric in ratio_signed_log_cols:
            # Skip target metric only for PERIOD=1
            if ratio_metric == target_metric and period == 1:
                continue
                
            as_is_key = f'as_is_{ratio_metric}_period_{period}'
            metric_series = group.get(ratio_metric)
            if metric_series is not None:
                feats[as_is_key] = metric_series.get(period, np.nan)
            else:
                feats[as_is_key] = np.nan
            numerical_feature_keys.append(as_is_key)

    feats['_numerical_feature_keys'] = list(set(numerical_feature_keys))
    return feats

class FeatureGenerator:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def generate_enhanced_features(
        self,
        df: pd.DataFrame,
        hist_window: int = 6,
        fwd_window: int = 6,
        target_metric: str = 'PE_RATIO_RATIO_SIGNED_LOG',
        sector_mapping_path: Optional[str] = None,
        sector_levels_to_include: List[str] = ['sector_1'],
        include_sector_features: bool = True
    ) -> pd.DataFrame:
        """
        Generates time series and optional sector features, including PERIOD=0 in hist/combined calculations.
        """
        self.logger.info("Starting enhanced feature generation...")

        # --- Input Validation and Setup ---
        required_cols = ['ID', 'PIT_DATE', 'PERIOD']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        if target_metric not in df.columns:
            self.logger.warning(f"Target metric '{target_metric}' not found in input DataFrame columns.")

        if hist_window < 0 or fwd_window < 0:
            raise ValueError("hist_window and fwd_window must be non-negative.")
        period_range = list(range(-hist_window, fwd_window + 1))
        self.logger.info(f"Using period range: {min(period_range)} to {max(period_range)} (inclusive)")

        # Identify metric columns dynamically
        raw_scaled_sales_signed_log_cols = [col for col in df.columns if '_RAW_SCALED_SALES_SIGNED_LOG' in col]
        
        # Get all ratio signed log columns, including both PE_RATIO and PREV_PE_RATIO
        ratio_signed_log_cols = []
        for col in df.columns:
            if '_RATIO_SIGNED_LOG' in col:
                # Only skip the target metric for PERIOD=1
                if col == target_metric and 'PERIOD=1' in col:
                    continue
                ratio_signed_log_cols.append(col)
                
        cstat_std_cols = [col for col in df.columns if '_CSTAT_STD' in col]
        
        # Debug logging
        self.logger.info(f"Input DataFrame columns containing RATIO_SIGNED_LOG: {[col for col in df.columns if '_RATIO_SIGNED_LOG' in col]}")
        self.logger.info(f"Selected ratio columns: {ratio_signed_log_cols}")
        
        self.logger.info(f"Identified metric columns: "
                        f"{len(raw_scaled_sales_signed_log_cols)} scaled sales, "
                        f"{len(ratio_signed_log_cols)} ratios, "
                        f"{len(cstat_std_cols)} stdevs.")

        # --- Core Feature Generation (Group Apply) ---
        global_duplicates = df.duplicated(subset=['ID', 'PIT_DATE', 'PERIOD']).sum()
        if global_duplicates > 0:
            self.logger.warning(f"Found {global_duplicates} duplicate rows in input based on (ID, PIT_DATE, PERIOD). Using first occurrence per group.")

        grouped = df.groupby(['ID', 'PIT_DATE'], observed=True, sort=False)
        n_groups = grouped.ngroups
        self.logger.info(f"Processing {n_groups} ID/PIT_DATE groups...")

        features_list = []
        group_iterator = tqdm(grouped, total=n_groups, desc="Processing groups") if tqdm else grouped

        for group_key, group_data in group_iterator:
            # Debug logging for first group
            if len(features_list) == 0:
                self.logger.info(f"First group columns: {group_data.columns.tolist()}")
                self.logger.info(f"First group ratio columns: {[col for col in group_data.columns if '_RATIO_SIGNED_LOG' in col]}")
            
            group_result = process_group(group_data.copy(), period_range,
                                      raw_scaled_sales_signed_log_cols,
                                      ratio_signed_log_cols,
                                      cstat_std_cols,
                                      target_metric,
                                      self.logger)
            
            # Debug logging for first group result
            if len(features_list) == 0 and group_result:
                self.logger.info(f"First group result keys: {list(group_result.keys())}")
                self.logger.info(f"First group result ratio features: {[k for k in group_result.keys() if '_RATIO_SIGNED_LOG' in k]}")
            
            if group_result:
                features_list.append(group_result)

        if not tqdm:
            self.logger.info(f"Finished processing {n_groups} groups.")

        if not features_list:
            self.logger.warning("No features generated. Returning empty DataFrame.")
            expected_cols = ['ID', 'PIT_DATE'] + ([target_metric] if target_metric in df.columns else [])
            return pd.DataFrame(columns=expected_cols)

        features_df = pd.DataFrame(features_list)
        
        # Debug logging for final features
        self.logger.info(f"Final features columns: {features_df.columns.tolist()}")
        self.logger.info(f"Final features ratio columns: {[col for col in features_df.columns if '_RATIO_SIGNED_LOG' in col]}")

        all_numerical_keys = set()
        if '_numerical_feature_keys' in features_df.columns:
            for keys_list in features_df['_numerical_feature_keys'].dropna():
                if isinstance(keys_list, list):
                    all_numerical_keys.update(keys_list)
            features_df = features_df.drop(columns=['_numerical_feature_keys'])
        numerical_feature_cols = [key for key in all_numerical_keys if key in features_df.columns]
        self.logger.info(f"Identified {len(numerical_feature_cols)} potential numerical feature columns generated.")

        # --- Optional Sector Feature Integration ---
        categorical_feature_names = [] # Keep track of added categorical features
        if include_sector_features and sector_mapping_path:
            try:
                self.logger.info(f"Loading sector mappings from: {sector_mapping_path}")
                if not os.path.exists(sector_mapping_path):
                    raise FileNotFoundError(f"Sector mapping file not found at {sector_mapping_path}")

                sector_df = pd.read_parquet(sector_mapping_path)
                valid_sector_levels = [col for col in sector_levels_to_include if col in sector_df.columns]
                if not valid_sector_levels:
                    self.logger.warning(f"None of specified sector levels {sector_levels_to_include} found in {sector_mapping_path}. Skipping sector processing.")
                else:
                    cols_to_merge = ['ID'] + valid_sector_levels
                    sector_df = sector_df[cols_to_merge].drop_duplicates(subset=['ID'], keep='first')
                    self.logger.info(f"Merging sector features for levels: {valid_sector_levels}")
                    original_feature_rows = len(features_df)
                    try:
                        # Attempt conversion before merge for consistency
                        id_dtype_feat = features_df['ID'].dtype
                        id_dtype_sect = sector_df['ID'].dtype
                        if id_dtype_feat != id_dtype_sect:
                           self.logger.warning(f"Mismatched ID dtypes ({id_dtype_feat} vs {id_dtype_sect}). Attempting cast to string for merge.")
                           features_df['ID'] = features_df['ID'].astype(str)
                           sector_df['ID'] = sector_df['ID'].astype(str)
                        features_df = features_df.merge(sector_df, on='ID', how='left', validate='m:1')
                        if len(features_df) != original_feature_rows:
                            self.logger.warning(f"Row count changed during sector merge ({original_feature_rows} -> {len(features_df)}). Check ID uniqueness.")
                    except Exception as merge_err:
                        self.logger.error(f"Error merging sectors: {merge_err}. Skipping sector processing.", exc_info=True)
                        valid_sector_levels = [] # Prevent further processing

                    # Process merged sector columns
                    sector_cols_in_features = [col for col in valid_sector_levels if col in features_df.columns]
                    if sector_cols_in_features:
                        fill_value = 'Missing_Sector' # Use a consistent string for missing
                        self.logger.info(f"Filling NaNs and converting to 'category' dtype for: {sector_cols_in_features}")
                        for col in sector_cols_in_features:
                            # Fill NaNs first
                            features_df[col] = features_df[col].fillna(fill_value)
                            # Convert to Pandas Categorical type
                            try:
                                features_df[col] = features_df[col].astype('category')
                                self.logger.debug(f"Column '{col}' converted to category. Categories: {features_df[col].cat.categories.tolist()}")
                            except Exception as cat_err:
                                 self.logger.error(f"Failed to convert column '{col}' to category dtype: {cat_err}. Skipping column.", exc_info=True)
                                 sector_cols_in_features.remove(col) # Remove if conversion failed
                        
                        categorical_feature_names.extend(sector_cols_in_features) # Store names of successfully converted cols
                        self.logger.info(f"Successfully added {len(categorical_feature_names)} categorical sector features.")
                        
                        # --- REMOVED OneHotEncoder LOGIC --- 
                        
            except FileNotFoundError as e:
                self.logger.warning(f"{e}. Skipping sector features.")
            except Exception as e:
                self.logger.warning(f"Error processing sector features: {e}. Skipping.", exc_info=True)
                categorical_feature_names = [] # Reset on error
        elif include_sector_features and not sector_mapping_path:
            self.logger.info("Sector features requested but no path provided. Skipping.")

        # --- Ranking (Only Numerical Features) ---
        processed_features_df = features_df.copy()
        # Ensure ranking only happens on numerical columns, excluding newly added categoricals
        numerical_cols_to_rank_candidates = [
            col for col in numerical_feature_cols 
            if col in processed_features_df.columns and col not in categorical_feature_names
        ]
        if not numerical_cols_to_rank_candidates:
            self.logger.warning("No numerical feature columns found to rank.")
        else:
            non_numeric_cols = processed_features_df[numerical_cols_to_rank_candidates].select_dtypes(exclude=[np.number]).columns
            if non_numeric_cols.any():
                self.logger.warning(f"Non-numeric columns found among numerical candidates for ranking: {non_numeric_cols.tolist()}. Excluding.")
                numerical_cols_to_rank = [col for col in numerical_cols_to_rank_candidates if col not in non_numeric_cols]
            else:
                numerical_cols_to_rank = numerical_cols_to_rank_candidates

            if not numerical_cols_to_rank:
                self.logger.info("Skipping ranking: No valid numeric columns remain.")
            else:
                self.logger.info(f"Ranking {len(numerical_cols_to_rank)} numerical features cross-sectionally (by PIT_DATE)...")
                try:
                    ranked_data = processed_features_df.groupby('PIT_DATE')[numerical_cols_to_rank].transform(lambda x: x.rank(pct=True))
                    rename_dict = {col: f'rank_{col}' for col in numerical_cols_to_rank}
                    ranked_data = ranked_data.rename(columns=rename_dict)
                    processed_features_df = processed_features_df.drop(columns=numerical_cols_to_rank)
                    processed_features_df = pd.concat([processed_features_df, ranked_data], axis=1)
                    self.logger.info("Numerical feature ranking complete.")
                except Exception as e:
                    self.logger.error(f"Error during ranking: {e}. Proceeding with unranked features.")

        # --- Merge Target Variable ---
        self.logger.info(f"Merging target variable: {target_metric}")
        if target_metric not in df.columns:
            self.logger.warning(f"Target metric '{target_metric}' not in original DataFrame. Cannot merge target.")
            if target_metric not in processed_features_df.columns:
                processed_features_df[target_metric] = np.nan
            final_df = processed_features_df
        else:
            # Get target value for PERIOD=1
            target_df = df[df['PERIOD'] == 1][['ID', 'PIT_DATE', target_metric]].drop_duplicates(subset=['ID', 'PIT_DATE'], keep='first')
            if target_df.empty:
                self.logger.warning(f"No data for PERIOD=1 to extract target '{target_metric}'. Target column will be all NaNs.")
                target_df_placeholder = processed_features_df[['ID', 'PIT_DATE']].drop_duplicates()
                target_df_placeholder[target_metric] = np.nan
                try:
                    final_df = processed_features_df.merge(target_df_placeholder, on=['ID', 'PIT_DATE'], how='left', validate='m:1')
                except Exception as merge_err:
                    self.logger.error(f"Error merging placeholder target: {merge_err}.")
                    if target_metric not in processed_features_df.columns:
                        processed_features_df[target_metric] = np.nan
                    final_df = processed_features_df
            else:
                try:
                    final_df = processed_features_df.merge(target_df, on=['ID', 'PIT_DATE'], how='left', validate='m:1')
                except Exception as merge_err:
                    self.logger.error(f"Error merging target: {merge_err}.")
                    if target_metric not in processed_features_df.columns:
                        processed_features_df[target_metric] = np.nan
                    final_df = processed_features_df

        if target_metric in final_df.columns:
            missing_target_fraction = final_df[target_metric].isnull().mean()
            if missing_target_fraction == 1.0:
                self.logger.warning(f"Target '{target_metric}' is ALL missing.")
            elif missing_target_fraction > 0:
                self.logger.warning(f"Target '{target_metric}' has {missing_target_fraction:.1%} missing values.")
        else:
            self.logger.warning(f"Target metric '{target_metric}' column not present in final DataFrame.")

        id_cols = ['ID', 'PIT_DATE']
        feature_cols = [col for col in final_df.columns if col not in id_cols and col != target_metric]
        final_cols_order = id_cols + sorted(feature_cols)
        if target_metric in final_df.columns:
            final_cols_order.append(target_metric)
        final_cols_order_exist = [col for col in final_cols_order if col in final_df.columns]
        try:
            final_df = final_df[final_cols_order_exist]
        except KeyError as e:
            self.logger.warning(f"Could not reorder columns: {e}")

        self.logger.info(f"Final DataFrame shape: {final_df.shape}")
        return final_df 