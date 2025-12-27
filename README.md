## Overview

A Monte Carlo simulation framework for optimizing Martingale betting strategies in blackjack with a focus on maximizing expected profit from daily freeroll opportunities. The system analyzes swing high patterns across parameter combinations to identify optimal cash-out points.

## Problem Statement

Given a $1 daily freeroll that must be wagered in blackjack, determine the optimal betting strategy that maximizes expected daily profit. Traditional Martingale analysis focuses on final bankroll, but this misses a critical insight: sessions often peak at higher values before declining. This project optimizes for swing highs rather than final outcomes.

## Methodology

### Core Approach

1. **Swing High Tracking**: Monitor peak bankroll reached during each session, not just final balance
2. **Expected Value Optimization**: For each parameter combination, calculate expected profit as (probability of reaching threshold) × (profit at threshold)
3. **Comprehensive Grid Search**: Test 1000+ parameter combinations to find global optimum

### Parameters Tested

- Starting bet size: 1% to 20% of bankroll
- Maximum consecutive losses before reset: 2 to 6
- Maximum hands per session: 100 to 2000
- Cash-out thresholds: $1.25 to $20.00

### Simulation Details

- 2000 Monte Carlo iterations per parameter combination
- Blackjack win probability: 49% (basic strategy)
- Table limits: $0.10 minimum, $500 maximum
- Martingale doubling with loss-limit circuit breaker

## Results

### Optimal Strategy

| Parameter | Value |
|-----------|-------|
| Starting Bet | 12% ($0.12) |
| Max Consecutive Losses | 3 |
| Max Hands | 750 |
| Cash-Out Target | $4.00 |
| Expected Daily Profit | $0.567 |
| Expected Annual Profit | $207 |
| Success Rate | 18.9% |

### Key Findings

1. **Swing High Analysis is Critical**: Optimal cash-out point ($4.00) would be missed by analyzing only final bankrolls
2. **Loss Limits Reduce Bust Rate**: Capping consecutive losses at 3 prevents catastrophic drawdowns while maintaining profitability
3. **Session Length Matters**: 750 hands provides sufficient opportunity to reach higher targets without excessive house edge exposure
4. **Conservative Betting Underperforms**: 1-5% starting bets have lower expected value despite higher success rates

## Usage

### Requirements
```python
numpy
pandas
matplotlib
```

### Running the Analysis

Execute cells sequentially in a Jupyter notebook:
```python
# Cell 1: Import libraries and set parameters
# Cell 2: Define simulation functions
# Cell 3: Run grid search across all combinations
# Cell 4: Calculate optimal cash-out points and rank strategies
# Cell 5: Display detailed breakdown and visualizations
```

### Customization

Adjust parameters in Cell 1:
```python
STARTING_BET_PCT_RANGE = [1, 2, 3, ..., 20]
MAX_CONSECUTIVE_LOSSES_RANGE = [2, 3, 4, 5, 6]
HANDS_PER_SIMULATION_RANGE = [100, 200, 300, ..., 2000]
NUM_SIMULATIONS = 2000
```

## Technical Details

### Swing High Calculation

For each session, track maximum bankroll reached:
```python
peak_bankroll = max(bankroll at each hand)
```

For each threshold T, calculate hit rate:
```python
hit_rate(T) = P(peak_bankroll >= T)
```

Expected profit at threshold T:
```python
expected_profit(T) = (T - starting_capital) × hit_rate(T)
```

Optimal threshold = argmax(expected_profit(T))

### Loss Limit Logic

After N consecutive losses, reset to base bet instead of continuing to double:
```python
if consecutive_losses >= max_consecutive_losses:
    current_bet = base_bet
    consecutive_losses = 0
else:
    current_bet = min(current_bet × 2, table_max)
```

## Interpretation

The strategy capitalizes on positive variance while limiting downside risk:

- 18.9% of sessions reach the $4.00 target
- 81.1% of sessions bust out (but cost nothing - freeroll)
- Expected value remains positive due to asymmetric payoff

Over 365 days:
- Approximately 69 successful sessions
- Average profit per success: $3.00
- Total expected profit: $207

## Limitations

1. Assumes 49% win rate (basic strategy) - card counting could improve this
2. Does not account for variance in actual play
3. Requires discipline to cash out at predetermined threshold
4. Table limits may prevent full doubling sequence
5. Freeroll must be wagerable (cannot simply withdraw $1)

## Future Work

- Incorporate actual blackjack strategy decisions beyond win probability
- Add Kelly Criterion-based bet sizing
- Test against historical casino data
- Optimize for risk-adjusted metrics (Sharpe ratio, Sortino ratio)
- Analyze correlation between parameters
