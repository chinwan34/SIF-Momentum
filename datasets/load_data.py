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

fundq = conn.raw_sql(f"""
    SELECT gvkey,  datadate,tic,
           epspxq, saleq, atq, actq,rectq, invtq, ceqq, lctq,
           cheq, cshoq, revty, dvpspq, dlttq, dlcq, teqq, ibq
    FROM comp.fundq
    WHERE datafmt='STD'
    AND datadate BETWEEN '1980-01-01' AND CURRENT_DATE
    AND gvkey IN ({','.join(f"'{g}'" for g in sp500_gvkeys)})
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
company_info = conn.raw_sql("""
    SELECT gvkey, gsector
    FROM comp.company
""")
fundq = fundq.merge(company_info, on='gvkey', how='left')
sec_dprc = conn.raw_sql(f"""
    SELECT gvkey,  datadate,ajexdi,
           prccd, eps
    FROM comp.sec_dprc
    WHERE gvkey IN ({','.join(f"'{g}'" for g in sp500_gvkeys)})
    AND datadate BETWEEN '1980-01-01' AND CURRENT_DATE
""")

sec_dprc['datadate'] = pd.to_datetime(sec_dprc['datadate'])
sec_dprc['qdate'] = sec_dprc['datadate'] + pd.tseries.offsets.QuarterEnd(0)
sec_dprc['qgap'] = (sec_dprc['qdate'] - sec_dprc['datadate']).abs()
sec_dprc = sec_dprc.sort_values(['qdate','qgap']).drop_duplicates(subset=['gvkey', 'qdate'], keep='first')
sec_dprc.drop(columns='qgap', inplace=True)
sec_dprc['adj_close_q'] = sec_dprc['prccd']/sec_dprc['ajexdi']

fundq['qdate'] = fundq['datadate'] + pd.tseries.offsets.QuarterEnd(0)
merged = pd.merge(
    fundq,
    sec_dprc[['gvkey','qdate','eps','adj_close_q']],
    on=['gvkey', 'qdate'],
    how='left'
)
merged

# Step 5: Get CRSP monthly prices
permnos = merged['permno'].dropna().unique().tolist()
msf = conn.raw_sql(f"""
    SELECT permno, date, ret
    FROM crsp.msf
    WHERE permno IN ({','.join(str(int(p)) for p in permnos)})
      AND date BETWEEN '1980-01-01' AND CURRENT_DATE
""")
msf['date'] = pd.to_datetime(msf['date'])
msf['qdate'] = msf['date'] + pd.tseries.offsets.QuarterEnd(0)
msf['qgap'] = (msf['qdate'] - msf['date']).abs()
msf = msf.sort_values(['qdate','qgap']).drop_duplicates(subset=['permno', 'qdate'], keep='first')
msf.drop(columns='qgap', inplace=True)

merged = pd.merge(
    merged,
    msf,
    on=['permno', 'qdate'],
    how='left'
)

merged['bps'] = merged['ceqq']/merged['cshoq']
merged['pe'] = merged['adj_close_q'] / merged['eps']
merged['pb'] = merged['adj_close_q'] / merged['bps']
merged['ps'] = merged['adj_close_q'] / (merged['revty'] / merged['cshoq'])

merged['cur_ratio'] = merged['actq'] / merged['lctq']
merged['quick_ratio'] = (merged['actq'] - merged['invtq']) / merged['lctq']
merged['debt_ratio'] = merged['dlttq'] / merged['atq']
merged['cash_ratio'] = merged['cheq'] / merged['lctq']
merged['acc_rec_turnover'] = merged['saleq'] / ((merged['rectq'] +merged['rectq'].shift(1))/2)
merged['debt_to_equity'] = (merged['dlttq'] + merged['dlcq']) / merged['teqq']

final = merged[[
    'datadate', 'gvkey', 'tic', 'adj_close_q', 'ret',
    'eps', 'bps', 'dvpspq', 'cur_ratio', 'quick_ratio', 'cash_ratio',
    'acc_rec_turnover', 'debt_ratio', 'debt_to_equity',
    'pe', 'ps', 'pb'
]]
final.rename(columns={'datadate': 'date', 'ret':'y_return','dvpspq':'dps'}, inplace=True)
final_ratios = pd.read_csv(r"datasets\final_ratios.csv")
final_ratios['gvkey']=final_ratios['gvkey'].apply(str)
final_ratios['date']=pd.to_datetime(final_ratios['date'])
final = pd.concat([final_ratios, final])
final=final.drop_duplicates(subset=['gvkey','date'],keep='first').sort_values(by=['gvkey','date'])
final.to_csv('final_ratios.csv')