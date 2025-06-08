import wrds
import pandas as pd

conn = wrds.Connection(wrds_username="sr2224")

today = pd.Timestamp.today().strftime('%m/%d/%Y')
sp500 = conn.raw_sql(f"""
    SELECT permno
    FROM crsp_a_indexes.dsp500list_v2
    WHERE NOT (mbrenddt < '{today}')
""")
permnos = tuple(sp500['permno'])

link_q = f"""
    SELECT DISTINCT l.permno, l.gvkey
    FROM crsp.ccmxpf_linktable l
    WHERE l.permno IN {permnos}
      AND l.linktype IN ('LU', 'LC')
      AND CURRENT_DATE BETWEEN l.linkdt AND COALESCE(l.linkenddt, CURRENT_DATE)
"""
permno_gvkey = conn.raw_sql(link_q)

sp500_gvkeys = permno_gvkey['gvkey'].unique().tolist()

fundq_vars = [
    'gvkey', 'datadate', 'tic', 'epspxq', 'ceqq', 'dvpsxq',
    'curcdq', 'quickrq', 'chratq', 'rectrq', 'dlttq', 'debtq',
    'saleq', 'cshoq'
]

fundq = conn.raw_sql(f"""
    SELECT {','.join(fundq_vars)}
    FROM comp.fundq
    WHERE gvkey IN ({','.join(f"'{g}'" for g in sp500_gvkeys)})
      AND datadate >= '2000-01-01'
""")

fundq['datadate'] = pd.to_datetime(fundq['datadate'])

ccm = conn.raw_sql("""
    SELECT gvkey, lpermno AS permno, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU', 'LC') AND usedflag = 1
""")
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.Timestamp('today'))

fundq = pd.merge(fundq, ccm, on='gvkey', how='left')
fundq = fundq[(fundq['datadate'] >= fundq['linkdt']) & (fundq['datadate'] <= fundq['linkenddt'])]
fundq = fundq.drop(columns=['linkdt', 'linkenddt'])

permnos_used = fundq['permno'].dropna().unique().tolist()
msf = conn.raw_sql(f"""
    SELECT permno, date, prc, ret
    FROM crsp.msf
    WHERE permno IN ({','.join(str(int(p)) for p in permnos_used)})
      AND date >= '2000-01-01'
""")
msf['date'] = pd.to_datetime(msf['date'])

fundq['qdate'] = fundq['datadate'] + pd.tseries.offsets.QuarterEnd(0)
msf['qdate'] = msf['date'] + pd.tseries.offsets.QuarterEnd(0)

merged = pd.merge(
    fundq,
    msf[['permno', 'qdate', 'prc', 'ret']],
    on=['permno', 'qdate'],
    how='left'
)

merged.rename(columns={
    'epspxq': 'EPS',
    'ceqq': 'BPS',
    'dvpsxq': 'DPS',
    'curcdq': 'cur_ratio',
    'quickrq': 'quick_ratio',
    'chratq': 'cash_ratio',
    'rectrq': 'acc_rec_turnover',
    'dlttq': 'debt_ratio',
    'debtq': 'debt_to_equity',
    'saleq': 'sales',
    'cshoq': 'shares_outstanding',
    'prc': 'adj_close_q',
    'ret': 'y_return'
}, inplace=True)

merged['pe'] = merged['adj_close_q'] / merged['EPS']
merged['pb'] = merged['adj_close_q'] / merged['BPS']
merged['ps'] = merged['adj_close_q'] / (merged['sales'] / merged['shares_outstanding'])

final = merged[[
    'datadate', 'gvkey', 'tic', 'adj_close_q', 'y_return',
    'EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio',
    'acc_rec_turnover', 'debt_ratio', 'debt_to_equity',
    'pe', 'ps', 'pb'
]]
final.rename(columns={'datadate': 'date'}, inplace=True)

final.to_csv('final_ratios.csv', index=False)
