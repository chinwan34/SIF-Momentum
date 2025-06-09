import wrds
import pandas as pd
import datetime

today = datetime.date.today().strftime('%Y-%m-%d')

conn = wrds.Connection(wrds_username="sr2224")

sp500 = conn.raw_sql(f"""
    SELECT permno
    FROM crsp_a_indexes.dsp500list_v2
    WHERE NOT (mbrenddt < DATE '2024-12-31')
""")
permnos = tuple(sp500['permno'])

link_q = f"""
    SELECT DISTINCT l.lpermno, l.gvkey
    FROM crsp.ccmxpf_linktable l
    WHERE l.lpermno IN {permnos}
      AND l.linktype IN ('LU', 'LC')
      AND CURRENT_DATE BETWEEN l.linkdt AND COALESCE(l.linkenddt, CURRENT_DATE)
"""
permno_gvkey = conn.raw_sql(link_q)

sp500_gvkeys = permno_gvkey['gvkey'].unique().tolist()

fundq_vars = [
    'gvkey', 'datadate', 'tic', 'epspxq', 'ceqq', 'dvpsxq',
    'curcdq', 'rectrq', 'dlttq', 'dlcq',
    'saleq', 'cshoq', 'actq', 'invtq', 'lctq', 'cheq'
]

fundq = conn.raw_sql(f"""
    SELECT {','.join(fundq_vars)}
    FROM comp.fundq
    WHERE gvkey IN ({','.join(f"'{g}'" for g in sp500_gvkeys)})
      AND datadate >= '2000-01-01'
""")
fundq['datadate'] = pd.to_datetime(fundq['datadate'])

link = conn.raw_sql("""
    SELECT gvkey, lpermno AS permno, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC') AND usedflag = 1
""")
link['linkdt'] = pd.to_datetime(link['linkdt'])
link['linkenddt'] = pd.to_datetime(link['linkenddt']).fillna(pd.Timestamp('today'))

fundq = pd.merge(fundq, link, on='gvkey', how='left')
fundq = fundq[(fundq['datadate'] >= fundq['linkdt']) & (fundq['datadate'] <= fundq['linkenddt'])]
fundq = fundq.drop(columns=['linkdt', 'linkenddt'])

# Step 5: Get CRSP monthly prices
permnos = fundq['permno'].dropna().unique().tolist()
msf = conn.raw_sql(f"""
    SELECT permno, date, prc, ret
    FROM crsp.msf
    WHERE permno IN ({','.join(str(int(p)) for p in permnos)})
      AND date >= '2000-01-01'
""")
msf['date'] = pd.to_datetime(msf['date'])

fundq['qdate'] = fundq['datadate'] + pd.tseries.offsets.QuarterEnd(0)
msf['qdate'] = msf['date'] + pd.tseries.offsets.QuarterEnd(0)

# Step 7: Merge price data
merged = pd.merge(
    fundq,
    msf[['permno', 'qdate', 'prc', 'ret']],
    on=['permno', 'qdate'],
    how='left'
)

# Step 8: Rename fields
merged.rename(columns={
    'epspxq': 'EPS',
    'ceqq': 'BPS',
    'dvpsxq': 'DPS',
    'curcdq': 'cur_ratio',
    'rectrq': 'acc_rec_turnover',
    'dlttq': 'debt_ratio',
    'prc': 'adj_close_q',
    'ret': 'y_return',
    'saleq': 'sales',
    'cshoq': 'shares_outstanding',
    'actq': 'current_assets',
    'invtq': 'inventory',
    'lctq': 'current_liabilities',
    'cheq': 'cash',
    'dlcq': 'short_term_debt'
}, inplace=True)

# Step 9: Compute all ratios
merged['pe'] = merged['adj_close_q'] / merged['EPS']
merged['pb'] = merged['adj_close_q'] / merged['BPS']
merged['ps'] = merged['adj_close_q'] / (merged['sales'] / merged['shares_outstanding'])
merged['quick_ratio'] = (merged['current_assets'] - merged['inventory']) / merged['current_liabilities']
merged['cash_ratio'] = merged['cash'] / merged['current_liabilities']
merged['total_debt'] = merged['debt_ratio'] + merged['short_term_debt']
merged['debt_to_equity'] = merged['total_debt'] / merged['BPS']

# Step 10: Final output
final = merged[[
    'datadate', 'gvkey', 'tic', 'adj_close_q', 'y_return',
    'EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio',
    'acc_rec_turnover', 'debt_ratio', 'debt_to_equity',
    'pe', 'ps', 'pb'
]]
final.rename(columns={'datadate': 'date'}, inplace=True)
final.to_csv('final_ratios.csv', index=False)