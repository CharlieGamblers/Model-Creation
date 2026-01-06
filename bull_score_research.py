import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Bootstrap for imports
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def analyze_tough_bull_performance():

    print("=== Tough Bull Performance Analysis ===")
       # Load batch predictions to identify toughest bulls
    print("\n1. Loading batch predictions to identify toughest bulls...")
    try:
        batch_preds = pd.read_excel("batch_predictions.xlsx")
        print(f"   Loaded {len(batch_preds)} predictions")
        print(f"   Columns: {batch_preds.columns.tolist()}")
    except FileNotFoundError:
        print("   ERROR: batch_predictions.xlsx not found!")
        return
    except PermissionError:
        print("   ERROR: batch_predictions.xlsx is open in Excel. Please close it and try again.")
        return
    
    # Check for required columns
    bull_col = None
    prob_col = None
    delivery_col = None
    
    for col in batch_preds.columns:
        if 'bull' in col.lower():
            bull_col = col
        if 'prob' in col.lower():
            prob_col = col
        if 'delivery' in col.lower():
            delivery_col = col
    
    # If Bull and Delivery are flipped, detect by checking for L/R patterns
    if bull_col and delivery_col:
        # Check if "Bull" column contains L/R (riding hand) and "Delivery" contains bull names
        bull_has_lr = batch_preds[bull_col].astype(str).str.contains(r'^[LR]$', na=False).any()
        delivery_has_lr = batch_preds[delivery_col].astype(str).str.contains(r'^[LR]$', na=False).any()
        
        if bull_has_lr and not delivery_has_lr:
            print(f"   Detected flipped columns: {bull_col} contains riding hands, {delivery_col} contains bulls")
            bull_col, delivery_col = delivery_col, bull_col
            print(f"   Swapped: Bull='{bull_col}', Delivery='{delivery_col}'")
    
    if bull_col is None or prob_col is None:
        print(f"   ERROR: Could not find Bull and Probability columns!")
        print(f"   Available columns: {batch_preds.columns.tolist()}")
        return
    
    print(f"   Using columns: Bull='{bull_col}', Probability='{prob_col}'")
    
    # Get unique bulls and their average probabilities (lower = tougher)
    bull_performance = batch_preds.groupby(bull_col)[prob_col].agg(['mean', 'count']).reset_index()
    bull_performance = bull_performance[bull_performance['count'] >= 3]  # At least 3 predictions
    
    # Sort by mean probability (ascending = toughest first)
    bull_performance = bull_performance.sort_values('mean')
    
    # Get toughest 20%
    n_tough = max(1, int(len(bull_performance) * 0.3))
    toughest_bulls = bull_performance.head(n_tough)[bull_col].tolist()
    
    print(f"   Identified {len(toughest_bulls)} toughest bulls (20% of {len(bull_performance)} total)")
    print(f"   Average probability range: {bull_performance['mean'].min():.1f}% - {bull_performance['mean'].max():.1f}%")
    print(f"   Toughest bulls: {', '.join(toughest_bulls[:5])}{'...' if len(toughest_bulls) > 5 else ''}")
    
    # Load base data
    print("\n2. Loading base data...")
    try:
        base_data = pd.read_csv("Data/Raw/base_data.csv")
        print(f"   Loaded {len(base_data):,} rides")
    except FileNotFoundError:
        print("   ERROR: Data/Raw/base_data.csv not found!")
        return
    
    # Create bull identifier (stock_no + stock_name)
    base_data['bull_identifier'] = (base_data['stock_no'].astype(str) + ' ' + 
                                   base_data['stock_name'].astype(str)).str.strip()
    
    # Filter for rides against toughest bulls
    print("\n3. Filtering rides against toughest bulls...")
    tough_bull_rides = base_data[base_data['bull_identifier'].isin(toughest_bulls)]
    print(f"   Found {len(tough_bull_rides):,} rides against toughest bulls")
    
    # Filter out invalid rides (RR, TO in comments, or stock_score = 0)
    print("\n4. Filtering out invalid rides...")
    initial_count = len(tough_bull_rides)
    
    # Filter out rides with RR or TO in comments
    if 'comments' in tough_bull_rides.columns:
        invalid_comments = tough_bull_rides['comments'].astype(str).str.contains(r'\b(RR|TO)\b', na=False, case=False)
        tough_bull_rides = tough_bull_rides[~invalid_comments]
        print(f"   Removed {invalid_comments.sum():,} rides with RR/TO in comments")
    
    # Filter out rides with stock_score = 0
    if 'stock_score' in tough_bull_rides.columns:
        zero_score = (tough_bull_rides['stock_score'] == 0) | (tough_bull_rides['stock_score'].isna())
        tough_bull_rides = tough_bull_rides[~zero_score]
        print(f"   Removed {zero_score.sum():,} rides with stock_score = 0 or missing")
    
    print(f"   Final dataset: {len(tough_bull_rides):,} valid rides (removed {initial_count - len(tough_bull_rides):,} total)")
    
    if len(tough_bull_rides) == 0:
        print("   ERROR: No rides found against toughest bulls!")
        print("   This might be due to naming differences between files.")
        return
    
    # Calculate rider performance against toughest bulls
    print("\n5. Calculating rider performance...")
    rider_performance = tough_bull_rides.groupby('rider').agg({
        'qr': ['count', 'sum', 'mean'],
        'ride_score': 'mean',
        'stock_score': 'mean'
    }).round(3)
    
    # Flatten column names
    rider_performance.columns = ['total_rides', 'qualified_rides', 'success_rate', 'avg_rider_score', 'avg_bull_score']
    rider_performance = rider_performance.reset_index()
    
    # Filter riders with at least 5 rides against tough bulls
    rider_performance = rider_performance[rider_performance['total_rides'] >= 5]
    rider_performance = rider_performance.sort_values('success_rate', ascending=False)
    
    print(f"   Found {len(rider_performance)} riders with 5+ rides against toughest bulls")
    
    # Display results
    print("\n=== TOP PERFORMERS AGAINST TOUGHEST BULLS ===")
    print("(Minimum 5 rides against toughest 20% of bulls)")
    print()
    print(f"{'Rider':<25} {'Rides':<6} {'Success%':<9} {'Avg Score':<10} {'Avg Bull Score':<12}")
    print("-" * 70)
    
    for _, row in rider_performance.head(15).iterrows():
        print(f"{row['rider']:<25} {row['total_rides']:<6} {row['success_rate']*100:<8.1f}% "
              f"{row['avg_rider_score']:<9.1f} {row['avg_bull_score']:<11.1f}")
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total rides analyzed: {len(tough_bull_rides):,}")
    print(f"Overall success rate vs tough bulls: {tough_bull_rides['qr'].mean()*100:.1f}%")
    print(f"Average rider score vs tough bulls: {tough_bull_rides['ride_score'].mean():.1f}")
    print(f"Average bull score (tough bulls): {tough_bull_rides['stock_score'].mean():.1f}")
    
    # Save detailed results
    output_file = "tough_bull_analysis.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        rider_performance.to_excel(writer, sheet_name='Rider Performance', index=False)
        toughest_bulls_df = pd.DataFrame({'Bull': toughest_bulls})
        toughest_bulls_df.to_excel(writer, sheet_name='Toughest Bulls', index=False)
        tough_bull_rides.to_excel(writer, sheet_name='Tough Bull Rides', index=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Additional analysis: High-scoring vs Low-scoring bulls
    print(f"\n=== ADDITIONAL ANALYSIS: HIGH vs LOW SCORING BULLS ===")
    analyze_bull_score_performance(base_data)
    
    return rider_performance, toughest_bulls, tough_bull_rides

def analyze_bull_score_performance(base_data):
    """
    Analyze rider performance on high-scoring bulls (43+) vs low-scoring bulls (43 and below).
    Only looks at rides from the last 5 years.
    """
    print("\n=== BULL SCORE PERFORMANCE ANALYSIS ===")
    
    # Filter to last 5 years first
    print("\n1. Filtering to last 5 years...")
    base_data['event_start_date'] = pd.to_datetime(base_data['event_start_date'])
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=5)
    base_data = base_data[base_data['event_start_date'] >= cutoff_date]
    print(f"   Filtered to {len(base_data):,} rides from last 5 years (since {cutoff_date.strftime('%Y-%m-%d')})")
    
    # Filter out invalid rides (same rules as tough bull analysis)
    print("\n2. Filtering out invalid rides...")
    initial_count = len(base_data)
    
    # Filter to PBR events only (excluding PBR Canada)
    if 'sanction' in base_data.columns:
        pbr_rides = base_data['sanction'] == 'PBR'
        base_data = base_data[pbr_rides]
        print(f"   Filtered to PBR events only: {len(base_data):,} rides")
        
        # Filter out PBR Canada rides
        if 'event_id' in base_data.columns:
            canada_rides = base_data['event_id'].astype(str).str.contains('CAN', case=False, na=False)
            base_data = base_data[~canada_rides]
            print(f"   Excluded PBR Canada rides: {canada_rides.sum():,} rides removed")
    else:
        print(f"   Warning: No 'sanction' column found, analyzing all rides")
    
    # Filter out rides with RR or TO in comments
    if 'comments' in base_data.columns:
        invalid_comments = base_data['comments'].astype(str).str.contains(r'\b(RR|TO)\b', na=False, case=False)
        base_data = base_data[~invalid_comments]
        print(f"   Removed {invalid_comments.sum():,} rides with RR/TO in comments")
    
    # Filter out rides with stock_score = 0 or missing
    if 'stock_score' in base_data.columns:
        zero_score = (base_data['stock_score'] == 0) | (base_data['stock_score'].isna())
        base_data = base_data[~zero_score]
        print(f"   Removed {zero_score.sum():,} rides with stock_score = 0 or missing")
    
    print(f"   Final dataset: {len(base_data):,} valid rides (removed {initial_count - len(base_data):,} total)")
    
    if len(base_data) == 0:
        print("   ERROR: No valid rides found!")
        return
    
    # Split into high-scoring (42+) and low-scoring (42 and below) bulls
    high_score_bulls = base_data[base_data['stock_score'] >= 42.5]
    low_score_bulls = base_data[base_data['stock_score'] < 42.5]
    
    print(f"\n3. Bull score distribution:")
    print(f"   High-scoring bulls (42+): {len(high_score_bulls):,} rides")
    print(f"   Low-scoring bulls (<42): {len(low_score_bulls):,} rides")
    print(f"   Average stock score - High: {high_score_bulls['stock_score'].mean():.1f}")
    print(f"   Average stock score - Low: {low_score_bulls['stock_score'].mean():.1f}")
    
    # Calculate rider performance for each group
    print(f"\n4. Calculating rider performance by bull score...")
    
    # High-scoring bulls performance
    high_performance = high_score_bulls.groupby('rider').agg({
        'qr': ['count', 'sum', 'mean'],
        'ride_score': 'mean',
        'stock_score': 'mean'
    }).round(3)
    high_performance.columns = ['total_rides', 'qualified_rides', 'success_rate', 'avg_rider_score', 'avg_bull_score']
    high_performance = high_performance.reset_index()
    high_performance = high_performance[high_performance['total_rides'] >= 5]  # Min 5 rides
    high_performance = high_performance.sort_values('success_rate', ascending=False)
    
    # Low-scoring bulls performance
    low_performance = low_score_bulls.groupby('rider').agg({
        'qr': ['count', 'sum', 'mean'],
        'ride_score': 'mean',
        'stock_score': 'mean'
    }).round(3)
    low_performance.columns = ['total_rides', 'qualified_rides', 'success_rate', 'avg_rider_score', 'avg_bull_score']
    low_performance = low_performance.reset_index()
    low_performance = low_performance[low_performance['total_rides'] >= 5]  # Min 5 rides
    low_performance = low_performance.sort_values('success_rate', ascending=False)
    
    print(f"   Found {len(high_performance)} riders with 5+ rides on high-scoring bulls")
    print(f"   Found {len(low_performance)} riders with 5+ rides on low-scoring bulls")
    
    # Display results
    print(f"\n=== TOP PERFORMERS ON HIGH-SCORING BULLS (42+) ===")
    print(f"{'Rider':<25} {'Rides':<6} {'Success%':<9} {'Avg Score':<10} {'Avg Bull Score':<12}")
    print("-" * 70)
    for _, row in high_performance.head(10).iterrows():
        print(f"{row['rider']:<25} {row['total_rides']:<6} {row['success_rate']*100:<8.1f}% "
              f"{row['avg_rider_score']:<9.1f} {row['avg_bull_score']:<11.1f}")
    
    print(f"\n=== TOP PERFORMERS ON LOW-SCORING BULLS (<42) ===")
    print(f"{'Rider':<25} {'Rides':<6} {'Success%':<9} {'Avg Score':<10} {'Avg Bull Score':<12}")
    print("-" * 70)
    for _, row in low_performance.head(10).iterrows():
        print(f"{row['rider']:<25} {row['total_rides']:<6} {row['success_rate']*100:<8.1f}% "
              f"{row['avg_rider_score']:<9.1f} {row['avg_bull_score']:<11.1f}")
    
    # Summary comparison
    print(f"\n=== SUMMARY COMPARISON ===")
    print(f"High-scoring bulls (42+):")
    print(f"  Total rides: {len(high_score_bulls):,}")
    print(f"  Overall success rate: {high_score_bulls['qr'].mean()*100:.1f}%")
    print(f"  Average rider score: {high_score_bulls['ride_score'].mean():.1f}")
    print(f"  Average bull score: {high_score_bulls['stock_score'].mean():.1f}")
    
    print(f"\nLow-scoring bulls (<42):")
    print(f"  Total rides: {len(low_score_bulls):,}")
    print(f"  Overall success rate: {low_score_bulls['qr'].mean()*100:.1f}%")
    print(f"  Average rider score: {low_score_bulls['ride_score'].mean():.1f}")
    print(f"  Average bull score: {low_score_bulls['stock_score'].mean():.1f}")
    
    # Find riders who perform well on both
    common_riders = set(high_performance['rider']) & set(low_performance['rider'])
    if common_riders:
        print(f"\n=== RIDERS WITH 5+ RIDES ON BOTH HIGH AND LOW SCORING BULLS ===")
        comparison_data = []
        for rider in common_riders:
            high_row = high_performance[high_performance['rider'] == rider].iloc[0]
            low_row = low_performance[low_performance['rider'] == rider].iloc[0]
            comparison_data.append({
                'rider': rider,
                'high_rides': high_row['total_rides'],
                'high_success': high_row['success_rate'] * 100,
                'low_rides': low_row['total_rides'],
                'low_success': low_row['success_rate'] * 100,
                'success_diff': (high_row['success_rate'] - low_row['success_rate']) * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('success_diff', ascending=False)
        print(f"{'Rider':<25} {'High%':<8} {'Low%':<8} {'Diff':<8} {'High Rides':<10} {'Low Rides':<10}")
        print("-" * 80)
        for _, row in comparison_df.head(10).iterrows():
            print(f"{row['rider']:<25} {row['high_success']:<7.1f}% {row['low_success']:<7.1f}% "
                  f"{row['success_diff']:+6.1f}% {row['high_rides']:<9} {row['low_rides']:<9}")
    
    # Save additional analysis
    output_file = "bull_score_analysis.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        high_performance.to_excel(writer, sheet_name='High Score Bulls', index=False)
        low_performance.to_excel(writer, sheet_name='Low Score Bulls', index=False)
        if common_riders:
            comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        high_score_bulls.to_excel(writer, sheet_name='High Score Rides', index=False)
        low_score_bulls.to_excel(writer, sheet_name='Low Score Rides', index=False)
    
    print(f"\nBull score analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_tough_bull_performance()
