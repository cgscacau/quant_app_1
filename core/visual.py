import plotly.graph_objects as go
import pandas as pd

def price_candles(df: pd.DataFrame):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
    )])
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=10,b=20))
    return fig

def line_series(df: pd.DataFrame, y: str, title:str=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[y], mode='lines', name=y))
    fig.update_layout(title=title, height=300, margin=dict(l=20,r=20,t=30,b=20))
    return fig

def equity_curve(df_eq: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Equity'], mode='lines', name='Equity'))
    fig.update_layout(title='Equity Curve', height=300, margin=dict(l=20,r=20,t=30,b=20))
    return fig