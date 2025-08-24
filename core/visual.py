import plotly.graph_objects as go
import pandas as pd




def price_candles(df: pd.DataFrame):
fig = go.Figure(data=[go.Candlestick(
x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
)])
fig.update_layout(height=420, margin=dict(l=20, r=20, t=10, b=20))
return fig




def line_series(df: pd.DataFrame, y: str, title: str = ''):
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[y], mode='lines', name=y))
fig.update_layout(title=title, height=320, margin=dict(l=20, r=20, t=40, b=20))
return fig
