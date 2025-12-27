"""
Martingale Swing High Optimizer
Monte Carlo simulation for optimizing Martingale betting strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

# Parameters
STARTING_CAPITAL = 1.0
STARTING_BET_PCT_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
MAX_CONSECUTIVE_LOSSES_RANGE = [2, 3, 4, 5, 6]
HANDS_PER_SIMULATION_RANGE = [100, 200, 300, 400, 500, 600, 750, 1000, 1500, 2000]
NUM_SIMULATIONS = 2000
WIN_PROBABILITY = 0.49
TABLE_MIN = 0.10
TABLE_MAX = 500
SWING_HIGH_THRESHOLDS = [1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 7.50, 10.00, 15.00, 20.00]

def play_hand(win_prob=0.49):
    """Simulate single blackjack hand"""
    return 1 if np.random.random() < win_prob else -1

def simulate_session_with_peak(starting_capital, starting_bet_pct, max_hands, 
                                max_consecutive_losses, table_min, table_max, win_prob):
    """
    Simulate Martingale session tracking peak bankroll
    
    Returns dict with final_bankroll, peak_bankroll, hands, busted
    """
    bankroll = starting_capital
    peak_bankroll = starting_capital
    base_bet = max(starting_capital * starting_bet_pct, table_min)
    current_bet = base_bet
    hands = 0
    consecutive_losses = 0
    busted = False
    
    while hands < max_hands:
        if current_bet > bankroll:
            busted = True
            break
        
        result = play_hand(win_prob)
        hands += 1
        
        if result == 1:
            bankroll += current_bet
            current_bet = base_bet
            consecutive_losses = 0
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
        else:
            bankroll -= current_bet
            consecutive_losses += 1
            if consecutive_losses >= max_consecutive_losses:
                current_bet = base_bet
                consecutive_losses = 0
            else:
                current_bet = min(current_bet * 2, table_max)
    
    return {
        'final_bankroll': bankroll,
        'peak_bankroll': peak_bankroll,
        'hands': hands,
        'busted': busted
    }

def run_monte_carlo(num_simulations, starting_capital, starting_bet_pct, max_hands, 
                   max_consecutive_losses, table_min, table_max, win_prob, swing_thresholds):
    """
    Run Monte Carlo simulations and calculate swing high statistics
    
    Returns dict with bust_rate, avg_hands, and swing_high_rates for each threshold
    """
    results = []
    for i in range(num_simulations):
        result = simulate_session_with_peak(starting_capital, starting_bet_pct, max_hands, 
                                           max_consecutive_losses, table_min, table_max, win_prob)
        results.append(result)
    
    df = pd.DataFrame(results)
    swing_high_rates = {}
    for threshold in swing_thresholds:
        swing_high_rates[f'hit_${threshold:.2f}'] = (df['peak_bankroll'] >= threshold).mean()
    
    return {
        'bust_rate': df['busted'].mean(),
        'avg_hands': df['hands'].mean(),
        'swing_high_rates': swing_high_rates
    }

def grid_search():
    """Execute grid search across all parameter combinations"""
    print("Starting grid search...")
    results = []
    current = 0
    total = len(STARTING_BET_PCT_RANGE) * len(MAX_CONSECUTIVE_LOSSES_RANGE) * len(HANDS_PER_SIMULATION_RANGE)
    
    for bet_pct in STARTING_BET_PCT_RANGE:
        for max_loss in MAX_CONSECUTIVE_LOSSES_RANGE:
            for max_hands in HANDS_PER_SIMULATION_RANGE:
                current += 1
                if current % 25 == 0 or current == total:
                    print(f"{current}/{total} ({current/total*100:.1f}%) - bet={bet_pct}%, loss={max_loss}, hands={max_hands}")
                
                stats = run_monte_carlo(NUM_SIMULATIONS, STARTING_CAPITAL, bet_pct/100, max_hands, 
                                       max_loss, TABLE_MIN, TABLE_MAX, WIN_PROBABILITY, SWING_HIGH_THRESHOLDS)
                
                result_row = {
                    'starting_bet_pct': bet_pct,
                    'max_consecutive_losses': max_loss,
                    'max_hands': max_hands,
                    'bust_rate': stats['bust_rate'],
                    'avg_hands': stats['avg_hands']
                }
                for key, value in stats['swing_high_rates'].items():
                    result_row[key] = value
                results.append(result_row)
    
    print("Grid search complete")
    return pd.DataFrame(results)

def find_optimal_strategies(grid_results):
    """
    For each parameter combination, find optimal cash-out threshold
    
    Returns DataFrame sorted by expected daily profit
    """
    optimal_results = []
    
    for idx, row in grid_results.iterrows():
        best_expected = 0
        best_threshold = STARTING_CAPITAL
        best_hit_rate = 0
        
        for threshold in SWING_HIGH_THRESHOLDS:
            hit_rate_key = f'hit_${threshold:.2f}'
            if hit_rate_key in row:
                hit_rate = row[hit_rate_key]
                profit = threshold - STARTING_CAPITAL
                expected = profit * hit_rate
                if expected > best_expected:
                    best_expected = expected
                    best_threshold = threshold
                    best_hit_rate = hit_rate
        
        optimal_results.append({
            'starting_bet_pct': row['starting_bet_pct'],
            'max_consecutive_losses': row['max_consecutive_losses'],
            'max_hands': row['max_hands'],
            'optimal_cashout': best_threshold,
            'optimal_profit': best_threshold - STARTING_CAPITAL,
            'hit_rate': best_hit_rate,
            'expected_daily_profit': best_expected,
            'bust_rate': row['bust_rate']
        })
    
    return pd.DataFrame(optimal_results).sort_values('expected_daily_profit', ascending=False).reset_index(drop=True)

def display_results(optimal_strategies):
    """Display top strategies and best strategy details"""
    print("\nTop 30 Strategies:\n")
    display_df = optimal_strategies.head(30).copy()
    display_df['hit_rate'] = (display_df['hit_rate'] * 100).round(1)
    display_df['bust_rate'] = (display_df['bust_rate'] * 100).round(1)
    print(display_df.to_string(index=True))
    
    best = optimal_strategies.iloc[0]
    print(f"\nOptimal Strategy:")
    print(f"Bet: {best['starting_bet_pct']:.0f}%")
    print(f"Max Losses: {best['max_consecutive_losses']:.0f}")
    print(f"Max Hands: {best['max_hands']:.0f}")
    print(f"Cash Out: ${best['optimal_cashout']:.2f}")
    print(f"Expected Daily: ${best['expected_daily_profit']:.4f}")
    print(f"Expected Annual: ${best['expected_daily_profit']*365:.2f}")

def display_breakdown(grid_results, optimal_strategies):
    """Display swing high breakdown for best strategy"""
    best_bet = optimal_strategies.iloc[0]['starting_bet_pct']
    best_loss = optimal_strategies.iloc[0]['max_consecutive_losses']
    best_hands = optimal_strategies.iloc[0]['max_hands']
    
    best_row = grid_results[
        (grid_results['starting_bet_pct']==best_bet) & 
        (grid_results['max_consecutive_losses']==best_loss) & 
        (grid_results['max_hands']==best_hands)
    ].iloc[0]
    
    print(f"\nSwing High Breakdown (Bet={best_bet}%, Loss={best_loss}, Hands={best_hands}):\n")
    swing_data = []
    max_expected = 0
    optimal_idx = 0
    
    for i, threshold in enumerate(SWING_HIGH_THRESHOLDS):
        hit_rate_key = f'hit_${threshold:.2f}'
        if hit_rate_key in best_row:
            hit_rate = best_row[hit_rate_key]
            profit = threshold - STARTING_CAPITAL
            expected = profit * hit_rate
            swing_data.append({
                'Threshold': f'${threshold:.2f}',
                'Profit': f'${profit:.2f}',
                'Hit_Rate': f'{hit_rate*100:.1f}%',
                'Expected': f'${expected:.3f}'
            })
            if expected > max_expected:
                max_expected = expected
                optimal_idx = i
    
    print(pd.DataFrame(swing_data).to_string(index=False))
    print(f"\nOptimal cash-out: ${SWING_HIGH_THRESHOLDS[optimal_idx]:.2f} (Expected: ${max_expected:.3f})")
    
    return best_bet, best_loss, best_hands

def visualize_results(best_bet, best_loss, best_hands, optimal_strategies):
    """Generate visualization of results"""
    df_sim = pd.DataFrame([
        simulate_session_with_peak(STARTING_CAPITAL, best_bet/100, int(best_hands), 
                                  int(best_loss), TABLE_MIN, TABLE_MAX, WIN_PROBABILITY) 
        for _ in range(NUM_SIMULATIONS)
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Peak bankroll distribution
    axes[0].hist(df_sim['peak_bankroll'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(optimal_strategies.iloc[0]['optimal_cashout'], color='red', 
                   linestyle='--', linewidth=2, 
                   label=f"Optimal: ${optimal_strategies.iloc[0]['optimal_cashout']:.2f}")
    axes[0].set_xlabel('Peak Bankroll')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Expected profit by cash-out level
    expected_profits = []
    labels = []
    for threshold in SWING_HIGH_THRESHOLDS:
        if threshold <= df_sim['peak_bankroll'].max():
            hit_rate = (df_sim['peak_bankroll'] >= threshold).mean()
            expected_profits.append((threshold - STARTING_CAPITAL) * hit_rate)
            labels.append(f'${threshold:.2f}')
    
    axes[1].plot(range(len(labels)), expected_profits, marker='o', linewidth=2)
    max_idx = expected_profits.index(max(expected_profits))
    axes[1].scatter([max_idx], [expected_profits[max_idx]], color='red', s=200, zorder=5)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_xlabel('Cash-Out Level')
    axes[1].set_ylabel('Expected Profit')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'optimization_results.png'")
    plt.show()

def main():
    """Main execution function"""
    print(f"Total combinations: {len(STARTING_BET_PCT_RANGE) * len(MAX_CONSECUTIVE_LOSSES_RANGE) * len(HANDS_PER_SIMULATION_RANGE)}")
    print(f"Estimated runtime: {(len(STARTING_BET_PCT_RANGE) * len(MAX_CONSECUTIVE_LOSSES_RANGE) * len(HANDS_PER_SIMULATION_RANGE) * 2) / 60:.1f} minutes\n")
    
    # Execute grid search
    grid_results = grid_search()
    
    # Find optimal strategies
    optimal_strategies = find_optimal_strategies(grid_results)
    
    # Display results
    display_results(optimal_strategies)
    
    # Display breakdown
    best_bet, best_loss, best_hands = display_breakdown(grid_results, optimal_strategies)
    
    # Visualize
    visualize_results(best_bet, best_loss, best_hands, optimal_strategies)
    
    # Export results
    grid_results.to_csv('grid_results.csv', index=False)
    optimal_strategies.to_csv('optimal_strategies.csv', index=False)
    print("\nResults exported to grid_results.csv and optimal_strategies.csv")

if __name__ == "__main__":
    main()
