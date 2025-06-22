import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# ======================
# 1. Market Data Simulator (with Level 2 Order Book)
# ======================
class MarketSimulator:
    def __init__(self, ticker, base_price, volatility, daily_volume):
        self.ticker = ticker
        self.price = base_price
        self.volatility = volatility  # Daily volatility (%)
        self.daily_volume = daily_volume  # Average daily volume (shares)
        self.price_history = [base_price]
        self.time_history = [0]
        self.order_book = self._init_order_book()
        
    def _init_order_book(self):
        """Initialize 5-level order book"""
        return {
            'bid': [{'price': self.price * (1 - i*0.001), 'qty': np.random.randint(1000,5000)} for i in range(1,6)],
            'ask': [{'price': self.price * (1 + i*0.001), 'qty': np.random.randint(1000,5000)} for i in range(1,6)]
        }
    
    def update_market(self, executed_volume=0, minute=None):
        """Simulate market changes: price movement + order book update"""
        # Geometric Brownian Motion price movement
        delta = self.volatility / np.sqrt(252) * np.random.randn()
        self.price *= (1 + delta)
        self.price_history.append(self.price)
        
        if minute is not None:
            self.time_history.append(minute)
        
        # Update order book (randomly modify levels)
        for side in ['bid', 'ask']:
            for level in self.order_book[side]:
                level['qty'] = max(100, int(level['qty'] * np.random.uniform(0.8, 1.2)))
                level['price'] *= (1 + np.random.uniform(-0.0005, 0.0005))
        
        # Volume consumption
        remaining_vol = self.daily_volume * 0.2  # Assume 20% of daily volume per minute
        return max(0, remaining_vol - executed_volume)
    
    def visualize_order_book(self):
        """Visualize current order book"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract bid and ask data
        bid_prices = [level['price'] for level in self.order_book['bid']]
        bid_qtys = [level['qty'] for level in self.order_book['bid']]
        ask_prices = [level['price'] for level in self.order_book['ask']]
        ask_qtys = [level['qty'] for level in self.order_book['ask']]
        
        # Plot bids
        ax.barh(np.arange(len(bid_prices)), bid_qtys, 
                color='green', alpha=0.6, label='Bids')
        
        # Plot asks
        ax.barh(np.arange(len(ask_prices)) + 0.3, ask_qtys, 
                color='red', alpha=0.6, label='Asks')
        
        # Add price labels
        for i, price in enumerate(bid_prices):
            ax.text(bid_qtys[i] + 500, i, f'{price:.2f}', va='center')
        for i, price in enumerate(ask_prices):
            ax.text(ask_qtys[i] + 500, i + 0.3, f'{price:.2f}', va='center')
        
        ax.set_yticks(np.arange(len(bid_prices)) + 0.15)
        ax.set_yticklabels([f'Level {i+1}' for i in range(len(bid_prices))])
        ax.set_xlabel('Order Quantity')
        ax.set_title(f'{self.ticker} Order Book Structure')
        ax.legend()
        plt.tight_layout()
        plt.show()

# ======================
# 2. VWAP Algorithm Implementation
# ======================
def vwap_algorithm(target_shares, market_data, duration_min=240):
    """
    VWAP Algorithm Executor
    :param target_shares: Target shares to trade
    :param market_data: MarketSimulator instance
    :param duration_min: Total trading duration (minutes)
    """
    print(f"\n=== Starting VWAP Algorithm [{market_data.ticker}] ===")
    print(f"Target: {target_shares/10000:.1f}M shares | Duration: {duration_min} min")
    
    # Historical volume distribution prediction (simplified: normal distribution)
    volume_profile = [np.exp(-(x-120)**2/(2*60**2)) for x in range(duration_min)]
    volume_profile = [v/sum(volume_profile)*target_shares for v in volume_profile]
    
    executed = 0
    results = []
    for minute in range(1, duration_min + 1):
        # Get current market state
        current_vol = volume_profile[minute-1] * np.random.uniform(0.9, 1.1)  # Add 10% randomness
        remaining_vol = market_data.update_market(current_vol, minute)
        
        # Calculate order quantity (max 20% of level depth)
        bid_depth = sum(level['qty'] for level in market_data.order_book['bid'][:3])  # Top 3 bid levels
        ask_depth = sum(level['qty'] for level in market_data.order_book['ask'][:3])  # Top 3 ask levels
        max_order_qty = min(bid_depth, ask_depth) * 0.2
        
        # Execute order (buy example)
        order_qty = min(max_order_qty, current_vol, target_shares - executed)
        if order_qty > 0:
            # Simulate execution at best ask price
            execute_price = market_data.order_book['ask'][0]['price']
            executed += order_qty
            results.append({
                'time': minute,
                'price': execute_price,
                'shares': order_qty,
                'cum_shares': executed,
                'algo': 'VWAP'
            })
            print(f"Min {minute:03d}: Executed {order_qty/10000:.1f}M shares @ {execute_price:.2f} | Progress: {executed/target_shares*100:.1f}%")
    
    # Performance analysis
    total_value = sum(r['price']*r['shares'] for r in results)
    avg_price = total_value / executed
    market_vwap = sum(market_data.price_history) / len(market_data.price_history)
    print(f"Execution completed! Avg Price: {avg_price:.2f} | Market VWAP: {market_vwap:.2f} | Slippage: {(avg_price-market_vwap)/market_vwap*100:.2f}%")
    
    # Visualization
    plot_execution(results, "VWAP Algorithm", market_data)
    return results

# ======================
# 3. TWAP Algorithm Implementation
# ======================
def twap_algorithm(target_shares, market_data, duration_min=240, volatility_threshold=0.3):
    """
    TWAP Algorithm Executor (with volatility adjustment)
    :param volatility_threshold: Volatility threshold to reduce speed
    """
    print(f"\n=== Starting TWAP Algorithm [{market_data.ticker}] ===")
    print(f"Target: {target_shares/10000:.1f}M shares | Duration: {duration_min} min")
    
    base_order = target_shares / duration_min
    executed = 0
    results = []
    prev_price = market_data.price
    
    for minute in range(1, duration_min + 1):
        # Update market state
        remaining_vol = market_data.update_market(0, minute)
        
        # Volatility detection (1-min change)
        current_volatility = abs(market_data.price / prev_price - 1)
        prev_price = market_data.price
        
        # Dynamic order adjustment (slow down during high volatility)
        adjustment = 1.0
        if current_volatility > volatility_threshold / 100:  # Convert to decimal
            adjustment = 0.5
            print(f"Min {minute:03d}: High volatility ({current_volatility*100:.2f}%), reducing speed by 50%")
        
        order_qty = base_order * adjustment
        order_qty = min(order_qty, target_shares - executed, remaining_vol)
        
        if order_qty > 0:
            # Simulate execution at mid-price
            bid_top = market_data.order_book['bid'][0]['price']
            ask_top = market_data.order_book['ask'][0]['price']
            execute_price = (bid_top + ask_top) / 2
            executed += order_qty
            results.append({
                'time': minute,
                'price': execute_price,
                'shares': order_qty,
                'cum_shares': executed,
                'algo': 'TWAP'
            })
            print(f"Min {minute:03d}: Executed {order_qty/10000:.1f}M shares @ {execute_price:.2f} | Progress: {executed/target_shares*100:.1f}%")
    
    # Performance analysis
    total_value = sum(r['price']*r['shares'] for r in results)
    avg_price = total_value / executed
    market_avg = sum(market_data.price_history) / len(market_data.price_history)
    print(f"Execution completed! Avg Price: {avg_price:.2f} | Market Avg: {market_avg:.2f} | Slippage: {(avg_price-market_avg)/market_avg*100:.2f}%")
    
    # Visualization
    plot_execution(results, "TWAP Algorithm", market_data)
    return results

# ======================
# 4. Iceberg Algorithm Implementation
# ======================
def iceberg_algorithm(target_shares, market_data, visible_ratio=0.1, duration_min=240):
    """
    Iceberg Algorithm Executor
    :param visible_ratio: Visible portion ratio
    """
    print(f"\n=== Starting Iceberg Algorithm [{market_data.ticker}] ===")
    print(f"Target: {target_shares/10000:.1f}M shares | Visible Ratio: {visible_ratio*100:.0f}%")
    
    executed = 0
    hidden_shares = target_shares
    results = []
    order_id = 0
    
    for minute in range(1, duration_min + 1):
        # Update market state
        remaining_vol = market_data.update_market(0, minute)
        
        if hidden_shares <= 0:
            break
            
        # Calculate visible order quantity (max 10% of top 3 levels depth)
        visible_qty = min(
            sum(level['qty'] for level in market_data.order_book['ask'][:3]) * visible_ratio,
            hidden_shares * 0.01  # Max 1% of total per order
        )
        
        # Place visible order
        if visible_qty > 1000:  # Minimum order size
            order_id += 1
            print(f"Min {minute:03d}: Placing visible order #{order_id} {visible_qty/10000:.1f}M shares @ {market_data.order_book['ask'][0]['price']:.2f}")
            
            # Simulate partial execution
            execute_qty = min(visible_qty, remaining_vol * 0.8)  # Assume 80% fill rate
            execute_price = market_data.order_book['ask'][0]['price']
            executed += execute_qty
            hidden_shares -= execute_qty
            results.append({
                'time': minute,
                'price': execute_price,
                'shares': execute_qty,
                'cum_shares': executed,
                'algo': 'Iceberg'
            })
            print(f"    Executed {execute_qty/10000:.1f}M shares | Hidden remaining: {hidden_shares/10000:.1f}M shares")
    
    # Complete remaining with market order
    if hidden_shares > 0:
        execute_price = market_data.order_book['ask'][0]['price'] * 1.005  # Pay 0.5% premium
        executed += hidden_shares
        results.append({
            'time': duration_min,
            'price': execute_price,
            'shares': hidden_shares,
            'cum_shares': executed,
            'algo': 'Iceberg (Market)'
        })
        print(f"Switching to market order! Executed {hidden_shares/10000:.1f}M shares @ {execute_price:.2f}")
    
    # Performance analysis
    total_value = sum(r['price']*r['shares'] for r in results)
    avg_price = total_value / executed
    market_avg = sum(market_data.price_history) / len(market_data.price_history)
    hidden_percent = (target_shares - results[0]['shares'])/target_shares*100
    print(f"Execution completed! Avg Price: {avg_price:.2f} | Hidden %: {hidden_percent:.1f}% | Slippage: {(avg_price-market_avg)/market_avg*100:.2f}%")
    
    # Visualization
    plot_execution(results, "Iceberg Algorithm", market_data)
    return results

# ======================
# 5. Dark Pool Algorithm Implementation
# ======================
class DarkPool:
    def __init__(self, name, liquidity_factor):
        self.name = name
        self.liquidity_factor = liquidity_factor  # Liquidity coefficient
        self.orders = []  # Orders waiting for match
        
    def match_orders(self):
        """Attempt to match buy/sell orders (simplified random matching)"""
        matched = []
        buy_orders = [o for o in self.orders if o['side'] == 'buy']
        sell_orders = [o for o in self.orders if o['side'] == 'sell']
        
        while buy_orders and sell_orders:
            buy = buy_orders.pop(0)
            sell = sell_orders.pop(0)
            match_qty = min(buy['qty'], sell['qty'])
            match_price = (buy['price'] + sell['price']) / 2  # Mid-price execution
            
            # Record transaction
            matched.append({
                'buy_order': buy['id'],
                'sell_order': sell['id'],
                'qty': match_qty,
                'price': match_price
            })
            
            # Update remaining quantities
            buy['qty'] -= match_qty
            sell['qty'] -= match_qty
            if buy['qty'] > 0: buy_orders.insert(0, buy)
            if sell['qty'] > 0: sell_orders.insert(0, sell)
        
        return matched

def darkpool_algorithm(target_shares, market_data, dark_pools, duration_min=240):
    """
    Dark Pool Algorithm Executor (multi-pool routing)
    """
    print(f"\n=== Starting Dark Pool Algorithm [{market_data.ticker}] ===")
    print(f"Target: {target_shares/10000:.1f}M shares | Available Dark Pools: {', '.join(dp.name for dp in dark_pools)}")
    
    executed = 0
    results = []
    order_id = 0
    
    for minute in range(1, duration_min + 1):
        # Update market state
        market_data.update_market(0, minute)
        
        if executed >= target_shares:
            break
            
        # Create dark pool order (buy)
        order_qty = min(target_shares - executed, 50000)  # Max 50k shares per order
        order_id += 1
        order = {
            'id': order_id,
            'ticker': market_data.ticker,
            'side': 'buy',
            'qty': order_qty,
            'price': market_data.price * (1 + np.random.uniform(-0.005, 0.005))  # ±0.5% price tolerance
        }
        
        # Select dark pool (weighted by liquidity factor)
        pool_weights = [dp.liquidity_factor for dp in dark_pools]
        total_weight = sum(pool_weights)
        chosen_pool = np.random.choice(dark_pools, p=[w/total_weight for w in pool_weights])
        chosen_pool.orders.append(order)
        print(f"Min {minute:03d}: Order #{order_id} sent to {chosen_pool.name} | {order_qty/10000:.1f}M shares")
        
        # All dark pools attempt matching
        for pool in dark_pools:
            matches = pool.match_orders()
            for match in matches:
                # Record transaction
                executed += match['qty']
                results.append({
                    'time': minute,
                    'price': match['price'],
                    'shares': match['qty'],
                    'cum_shares': executed,
                    'pool': pool.name,
                    'algo': 'Dark Pool'
                })
                print(f"    Dark pool execution: {match['qty']/10000:.1f}M shares @ {match['price']:.2f} ({pool.name})")
    
    # Complete remaining in open market
    if executed < target_shares:
        remaining = target_shares - executed
        execute_price = market_data.order_book['ask'][0]['price'] * 1.01  # Pay 1% premium
        executed += remaining
        results.append({
            'time': duration_min,
            'price': execute_price,
            'shares': remaining,
            'cum_shares': executed,
            'pool': 'OPEN MARKET',
            'algo': 'Dark Pool (Market)'
        })
        print(f"Open market execution: {remaining/10000:.1f}M shares @ {execute_price:.2f}")
    
    # Performance analysis
    total_value = sum(r['price']*r['shares'] for r in results)
    avg_price = total_value / executed
    market_avg = sum(market_data.price_history) / len(market_data.price_history)
    dark_percent = sum(r['shares'] for r in results if r['pool'] != 'OPEN MARKET') / executed * 100
    print(f"Execution completed! Avg Price: {avg_price:.2f} | Dark %: {dark_percent:.1f}% | Slippage: {(avg_price-market_avg)/market_avg*100:.2f}%")
    
    # Visualization
    plot_execution(results, "Dark Pool Algorithm", market_data, dark_pools=[dp.name for dp in dark_pools])
    return results

# ======================
# Enhanced Visualization Function
# ======================
def plot_execution(results, title, market_data=None, dark_pools=None):
    """Visualize algorithm execution results with enhanced plots"""
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Create Plotly figure
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Execution Price vs Market Price', 
                                       'Cumulative Execution', 
                                       'Price Impact Analysis'),
                        row_heights=[0.4, 0.3, 0.3])
    
    # Price comparison plot
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['price'], 
        mode='markers+lines', name='Execution Price',
        marker=dict(size=8, color='blue'),
        line=dict(width=2)
    ), row=1, col=1)
    
    if market_data:
        fig.add_trace(go.Scatter(
            x=market_data.time_history, y=market_data.price_history,
            mode='lines', name='Market Price',
            line=dict(width=2, color='red')
        ), row=1, col=1)
    
    # Cumulative execution
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['cum_shares'],
        mode='lines+markers', name='Cumulative Shares',
        fill='tozeroy', fillcolor='rgba(0,128,0,0.2)',
        line=dict(width=3, color='green')
    ), row=2, col=1)
    
    # Price impact analysis
    if len(df) > 1:
        df['trade_size'] = df['shares'] / 10000  # Convert to 10k shares units
        df['price_impact'] = (df['price'] / market_data.price_history[0] - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['price_impact'],
            mode='lines+markers', name='Price Impact (%)',
            line=dict(width=2, color='purple')
        ), row=3, col=1)
        
        # Add regression line
        z = np.polyfit(df['time'], df['price_impact'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['time'], y=p(df['time']),
            mode='lines', name='Impact Trend',
            line=dict(width=2, dash='dash', color='orange')
        ), row=3, col=1)
    
    # Add dark pool annotations
    if dark_pools and 'pool' in df.columns:
        dark_trades = df[df['pool'] != 'OPEN MARKET']
        for _, row in dark_trades.iterrows():
            fig.add_annotation(
                x=row['time'], y=row['price'],
                text=row['pool'],
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-40,
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f'<b>{title} Performance Analysis</b>',
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text='Time (minutes)', row=3, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Shares', row=2, col=1)
    fig.update_yaxes(title_text='Price Impact (%)', row=3, col=1)
    
    fig.show()

# ======================
# Main Demo Execution
# ======================
if __name__ == "__main__":
    print("Starting Trading Algorithms Demonstration...")
    
    # Create simulated market environment for MSFT
    msft_market = MarketSimulator(ticker="MSFT", base_price=300, volatility=1.5, daily_volume=5000000)
    msft_market.visualize_order_book()
    
    # Execute VWAP algorithm (3M shares, 4 hours)
    vwap_results = vwap_algorithm(target_shares=3000000, market_data=msft_market)
    
    # Execute TWAP algorithm (with volatility adjustment)
    msft_market = MarketSimulator(ticker="MSFT", base_price=300, volatility=1.5, daily_volume=5000000)
    twap_results = twap_algorithm(target_shares=3000000, market_data=msft_market, volatility_threshold=0.4)
    
    # Execute Iceberg algorithm in high volatility environment
    msft_highvol = MarketSimulator(ticker="MSFT", base_price=300, volatility=3.0, daily_volume=5000000)
    iceberg_results = iceberg_algorithm(target_shares=3000000, market_data=msft_highvol, visible_ratio=0.1)
    
    # Set up dark pools
    dark_pools = [
        DarkPool(name="GS SigmaX", liquidity_factor=0.6),
        DarkPool(name="MS Pool", liquidity_factor=0.3),
        DarkPool(name="Liquidnet", liquidity_factor=0.1)
    ]
    
    # Execute Dark Pool algorithm
    msft_market = MarketSimulator(ticker="MSFT", base_price=300, volatility=1.5, daily_volume=5000000)
    darkpool_results = darkpool_algorithm(target_shares=3000000, market_data=msft_market, dark_pools=dark_pools)
    
    print("\n=== All Algorithm Simulations Completed ===")