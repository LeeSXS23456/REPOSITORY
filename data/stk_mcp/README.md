提取了自由流通股本和收盘价，计算自由流通市值

a = get_shares(stk, start_date=dt, end_date=dt, fields='free_circulation', expect_df=True, market='cn').droplevel(1)
    
b = get_price(stk, start_date=dt, end_date=dt, frequency='1d', fields="close", adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn').droplevel(1)

df = a.merge(b[["close"]],on="order_book_id",how="left")
