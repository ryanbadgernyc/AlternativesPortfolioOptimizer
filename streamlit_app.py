import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import cvxpy as cp
from scipy.spatial import ConvexHull

# Configure page
st.set_page_config(page_title="Efficient Frontier Simulator", layout="wide")
st.title('Efficient Frontier Simulator with Exact Optimization')

# --- Dynamic Asset Classes ---
asset_input = st.text_area(
    "Enter asset classes separated by commas:",
    value="Venture, Infrastructure, US Buyouts, SFR, CRE, Farmland",
    help="Type asset names separated by commas, e.g. 'Stocks, Bonds, Gold'"
)
assets = [a.strip() for a in asset_input.split(',') if a.strip()]

# --- Hardcoded Defaults ---
default_returns = {'Venture':12.5,'Infrastructure':9.5,'US Buyouts':9.5,'SFR':8.1,'CRE':6.6,'Farmland':10.1}
default_vols    = {'Venture':18.0,'Infrastructure':11.0,'US Buyouts':14.0,'SFR':7.0,'CRE':11.0,'Farmland':9.0}
default_max     = {'Venture':15.0,'Infrastructure':15.0,'US Buyouts':100.0,'SFR':15.0,'CRE':50.0,'Farmland':15.0}

# Correlation defaults
known_assets = ['Venture','Infrastructure','US Buyouts','SFR','CRE','Farmland']
default_corr = pd.DataFrame([
    [1.00,0.50,0.80,0.60,0.30,0.00],
    [0.50,1.00,0.65,0.40,0.40,0.00],
    [0.80,0.65,1.00,0.60,0.35,0.00],
    [0.60,0.40,0.60,1.00,0.35,0.20],
    [0.30,0.40,0.35,0.35,1.00,0.10],
    [0.00,0.00,0.00,0.20,0.10,1.00]
], index=known_assets, columns=known_assets)

# --- Asset Assumptions Inputs ---
st.subheader("Asset Assumptions")
returns, stds, max_weights = [], [], []
for asset in assets:
    c1, c2, c3 = st.columns(3)
    r = c1.slider(
        f"{asset} Return (%):", -100.0, 100.0,
        default_returns.get(asset,0.0), step=0.25,
        key=f"ret_{asset}"
    )
    v = c2.slider(
        f"{asset} Volatility (%):", 0.0, 100.0,
        default_vols.get(asset,0.0), step=0.25,
        key=f"vol_{asset}"
    )
    m = c3.slider(
        f"{asset} Max Weight (%):", 0.0, 100.0,
        default_max.get(asset,0.0), step=0.1,
        key=f"max_{asset}"
    )
    returns.append(r/100)
    stds.append(v/100)
    max_weights.append(m/100)
returns = np.array(returns)
stds    = np.array(stds)
max_weights = np.array(max_weights)

# --- Correlation Matrix Inputs ---
st.subheader("Correlation Matrix")
corr_df = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
for i, ai in enumerate(assets):
    for j, aj in enumerate(assets):
        if ai in known_assets and aj in known_assets:
            corr_df.iat[i,j] = default_corr.loc[ai,aj]

st.write("Enter pairwise correlations (only lower triangle editable):")
col_widths = [1] * (len(assets) + 1)
header_cols = st.columns(col_widths)
header_cols[0].markdown("**Asset**")
for idx, asset in enumerate(assets):
    header_cols[idx+1].markdown(f"**{asset}**")
for i, ai in enumerate(assets):
    row_cols = st.columns(col_widths)
    row_cols[0].markdown(f"**{ai}**")
    for j, aj in enumerate(assets):
        if j < i:
            default_val = corr_df.iat[i,j]
            val = row_cols[j+1].number_input(
                label="", min_value=-1.0, max_value=1.0,
                value=float(default_val), step=0.01,
                key=f"corr_{i}_{j}"
            )
            corr_df.iat[i,j] = val
            corr_df.iat[j,i] = val
        elif j == i:
            row_cols[j+1].number_input(label="", value=1.0, disabled=True, key=f"corr_{i}_{j}")
        else:
            row_cols[j+1].markdown("&nbsp;")

cov = np.outer(stds, stds) * corr_df.values

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
fig_hm, ax_hm = plt.subplots(figsize=(max(6,len(assets)), max(6,len(assets))))
cax = ax_hm.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax_hm.set_xticks(np.arange(len(assets)))
ax_hm.set_yticks(np.arange(len(assets)))
ax_hm.set_xticklabels(assets, rotation=45, ha='right')
ax_hm.set_yticklabels(assets)
for i in range(len(assets)):
    for j in range(len(assets)):
        ax_hm.text(j, i, f"{corr_df.iat[i,j]:.2f}", ha='center', va='center')
fig_hm.colorbar(cax, ax=ax_hm, shrink=0.8)
st.pyplot(fig_hm)

# --- Current Portfolio Input ---
st.subheader('Current Portfolio Weights')
default_current = {'Venture':10.0,'Infrastructure':10.0,'US Buyouts':40.0,'SFR':10.0,'CRE':30.0,'Farmland':0.0}

cw = np.array([
    st.number_input(
        f"{asset} Current Weight (%):",0.0,100.0,
        default_current.get(asset,0.0),step=0.1,
        key=f"cw_{asset}"
    )/100
    for asset in assets
])
sum_pct = cw.sum()*100
if abs(sum_pct-100.0)>1e-6:
    st.error(f"Current weights sum to {sum_pct:.1f}%. Please ensure they total 100%.")
    valid_current = False
else:
    valid_current = True
    cw = cw/cw.sum()

# --- Exact Efficient Frontier via CVXPY ---
st.subheader('Exact Efficient Frontier (CVXPY)')
n, mu, Sigma = len(assets), returns, cov
targets = np.linspace(mu.min(), mu.max(), 50)
vols, rets, wts = [], [], []
for t in targets:
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
                      [cp.sum(w)==1, mu@w>=t, w>=0, w<=max_weights])
    prob.solve(solver=cp.ECOS)
    if w.value is not None:
        rets.append(float(mu@w.value))
        vols.append(float(np.sqrt(w.value.T@Sigma@w.value)))
        wts.append(w.value)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(vols, rets, '-o', label='Efficient Frontier')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=1))
sr = np.array(rets)/np.array(vols)
ix = sr.argmax();bv,br = vols[ix], rets[ix]
ax.scatter(bv,br,color='red',s=100,label='Max Sharpe')
ax.annotate('Max Sharpe',(bv,br),textcoords='offset points',xytext=(0,10),ha='center',color='red')
if valid_current:
    ret0, vol0 = float(mu@cw), float(np.sqrt(cw.T@Sigma@cw))
    ax.scatter(vol0,ret0,color='blue',marker='X',s=100,label='Current')
    ax.annotate('Current',(vol0,ret0),textcoords='offset points',xytext=(0,-15),ha='center',color='blue')
ax.set_xlabel('Volatility');ax.set_ylabel('Expected Return');ax.legend()
st.pyplot(fig)

# --- Portfolio Weights Change Table ---
if valid_current:
    best_w = wts[ix]
    df_change = pd.DataFrame({'Asset':assets,'Current':cw,'Optimal':best_w})
    df_change['Change'] = df_change['Optimal'] - df_change['Current']
    def color_change(val):
        return 'color: green' if val>0 else ('color: red' if val<0 else '')
    styled = df_change.style.format({'Current':'{:.2%}','Optimal':'{:.2%}','Change':'{:+.2%}'}) 
    styled = styled.applymap(color_change, subset=['Change'])
    st.subheader('Portfolio Weight Changes to Max Sharpe')
    st.dataframe(styled)

# --- Export Efficient Frontier ---
ef_df = pd.DataFrame(np.column_stack([vols,rets,np.array(wts)]),columns=['Volatility','Expected Return']+assets)
ef_df['Volatility'] = ef_df['Volatility'].apply(lambda x:f"{x:.2%}")
ef_df['Expected Return']=ef_df['Expected Return'].apply(lambda x:f"{x:.2%}")
csv = ef_df.to_csv(index=False).encode('utf-8')
st.download_button('Download Efficient Frontier CSV',data=csv,file_name='efficient_frontier.csv',mime='text/csv')
