# -*- coding: utf-8 -*-
from enum import Enum
from typing import List, Union

import pandas as pd

from zvt import IntervalLevel, AdjustType
from zvt.api import get_kdata
from zvt.contract import EntityMixin
from zvt.domain import Stock
from zvt.factors import Transformer, TechnicalFactor, Accumulator
from zvt.utils import pd_is_not_null


class Direction(Enum):
    up = 'up'
    down = 'down'

    def opposite(self):
        if self == Direction.up:
            return Direction.down
        if self == Direction.down:
            return Direction.up


class KState(Enum):
    # 顶分型
    ding = 'ding'
    # 底分型
    di = 'di'
    # 临时
    tmp_ding = 'tmp_ding'
    tmp_di = 'tmp_di'
    # 候选(candidate)
    can_ding = 'can_ding'
    can_di = 'can_di'


def a_include_b(a: pd.Series, b: pd.Series) -> bool:
    """
    kdata a includes kdata b

    :param a:
    :param b:
    :return:
    """
    return (a['high'] >= b['high']) and (a['low'] <= b['low'])


def is_including(kdata1, kdata2):
    return a_include_b(kdata1, kdata2) or a_include_b(kdata2, kdata1)


def get_direction(kdata, pre_kdata, current=Direction.up) -> Direction:
    if is_up(kdata, pre_kdata):
        return Direction.up
    if is_down(kdata, pre_kdata):
        return Direction.down

    return current


def is_up(kdata, pre_kdata):
    return kdata['high'] > pre_kdata['high']


def is_keep(kdata, pre_kdata, direction):
    keep1 = (direction == 'up') and (kdata['high'] == pre_kdata['high'])
    keep2 = (direction == 'down') and (kdata['low'] == pre_kdata['low'])
    return keep1 or keep2


def is_down(kdata, pre_kdata):
    return kdata['low'] < pre_kdata['low']


def first_fenxing(one_df, step=11):
    print(f"gen firest fenxing by step {step}")
    df = one_df.iloc[:step]
    ding_kdata = df[df['high'].max() == df['high']]
    ding_index = ding_kdata.index[-1]

    di_kdata = df[df['low'].min() == df['low']]
    di_index = di_kdata.index[-1]

    # 确定第一个分型
    if abs(ding_index - di_index) >= 4:
        if ding_index > di_index:
            one_df.loc[di_index, 'di'] = True
            # 确定第一个分型后，开始遍历的位置
            start_index = ding_index
            # 目前的笔的方向，up代表寻找 can_ding;down代表寻找can_di
            direction = Direction.up
        else:
            one_df.loc[ding_index, 'ding'] = True
            start_index = di_index
            direction = Direction.down
        return (start_index, direction)
    else:
        print("need add step")
        first_fenxing(one_df, step=step + 1)


def hanlde_including(one_df, index, kdata, pre_index, pre_kdata, tmp_direction: Direction):
    # 改kdata
    if a_include_b(kdata, pre_kdata):
        # 长的kdata变短
        if tmp_direction == Direction.up:
            one_df.loc[index, 'low'] = pre_kdata['low']
        else:
            one_df.loc[index, 'high'] = pre_kdata['high']
    # 改pre_kdata
    elif a_include_b(pre_kdata, kdata):
        # 长的pre_kdata变短
        if tmp_direction == Direction.down:
            one_df.loc[pre_index, 'low'] = kdata['low']
        else:
            one_df.loc[pre_index, 'high'] = kdata['high']


class ZenTransformer(Transformer):
    def __init__(self) -> None:
        super().__init__()
        # 算法和概念
        # <实体> 某种状态的k线
        # [实体] 连续实体排列

        # 两k线的关系有三种: 上涨，下跌，包含
        # 上涨: k线高点比之前高，低点比之前高
        # 下跌: k线低点比之前低，高点比之前低
        # 包含: k线高点比之前高，低点比之前低;反方向，即被包含
        # 处理包含关系，长的k线缩短，上涨时，低点取max(low1,low2)；下跌时，高点取min(high1,high2)

        # 第一个顶(底)分型: 出现连续4根下跌(上涨)k线
        # 之后开始寻找 候选底(顶)分型，寻找的过程中有以下状态

        # <临时顶>: 中间k线比两边的高点高,是一条特定的k线
        # <临时底>: 中间k线比两边的高点高,是一条特定的k线

        # <候选顶分型>: 连续的<临时顶>取最大
        # <候选底分型>:  连续的<临时底>取最小
        # 任何时刻只能有一个候选，其之前是一个确定的分型

        # <上升k线>:
        # <下降k线>:
        # <连接k线>: 分型之间的k线都可以认为是连接k线，以上为演化过程的中间态
        # distance(<候选顶分型>, <连接k线>)>=4 则 <候选顶分型> 变成顶分型
        # distance(<候选底分型>, <连接k线>)>=4 则 <候选底分型> 变成底分型

        # <顶分型><连接k线><候选底分型>
        # <底分型><连接k线><候选顶分型>

    def transform_one(self, one_df: pd.DataFrame) -> pd.DataFrame:
        one_df = one_df.reset_index(drop=False)
        one_df['ding'] = False
        one_df['di'] = False
        # 记录临时分型，不变
        one_df['tmp_ding'] = False
        one_df['tmp_di'] = False

        # 取前11条k线，至多出现一个顶分型+底分型
        # 注:只是一种方便的确定第一个分型的办法，有了第一个分型，后面的处理就比较统一
        # start_index 为遍历开始的位置
        # direction为一个确定分型后的方向，即顶分型后为:down，底分型后为:up
        start_index, direction = first_fenxing(one_df, step=11)
        # 临时方向
        tmp_direction = direction
        # 候选分型(candidate)
        can_fenxing = None
        can_fenxing_index = None
        # 正向count
        count = 0
        # 反方向count
        opposite_count = 0

        pre_kdata = one_df.iloc[start_index - 1]
        pre_index = start_index - 1
        for index, kdata in one_df.iloc[start_index:].iterrows():
            print(f'timestamp: {kdata.timestamp}')
            # 临时方向
            tmp_direction = get_direction(kdata, pre_kdata, current=tmp_direction)

            # 处理包含关系
            hanlde_including(one_df=one_df, index=index, kdata=kdata, pre_index=pre_index, pre_kdata=pre_kdata,
                             tmp_direction=tmp_direction)

            # 根据方向，寻找对应的分型
            if direction == Direction.up:
                tmp_fenxing_col = 'tmp_ding'
                fenxing_col = 'ding'
            else:
                tmp_fenxing_col = 'tmp_di'
                fenxing_col = 'di'

            # 方向一致，延续中
            if tmp_direction == direction:
                opposite_count = 0
            # 反向，寻找反 分型
            else:
                opposite_count = opposite_count + 1
                # 第一次反向
                if opposite_count == 1:
                    one_df.loc[pre_index, tmp_fenxing_col] = True

                    if pd_is_not_null(can_fenxing):
                        # 候选底分型
                        if tmp_direction == Direction.up:
                            # 取小的
                            if pre_kdata['low'] <= can_fenxing['low']:
                                can_fenxing = pre_kdata
                                can_fenxing_index = pre_index

                        # 候选顶分型
                        else:
                            # 取大的
                            if pre_kdata['high'] >= can_fenxing['high']:
                                can_fenxing = pre_kdata
                                can_fenxing_index = pre_index
                    else:
                        can_fenxing = pre_kdata
                        can_fenxing_index = pre_index

                # 分型确立
                if pd_is_not_null(can_fenxing):
                    if opposite_count >= 4 or (index - can_fenxing_index >= 8):
                        one_df.loc[can_fenxing_index, fenxing_col] = True
                        opposite_count = 0
                        direction = direction.opposite()
                        can_fenxing = None

            pre_kdata = kdata
            pre_index = index

        return one_df


class ZenFactor(TechnicalFactor):

    def __init__(self, entity_schema: EntityMixin = Stock, provider: str = None, entity_provider: str = None,
                 entity_ids: List[str] = None, exchanges: List[str] = None, codes: List[str] = None,
                 the_timestamp: Union[str, pd.Timestamp] = None, start_timestamp: Union[str, pd.Timestamp] = None,
                 end_timestamp: Union[str, pd.Timestamp] = None, columns: List = None, filters: List = None,
                 order: object = None, limit: int = None, level: Union[str, IntervalLevel] = IntervalLevel.LEVEL_1DAY,
                 category_field: str = 'entity_id', time_field: str = 'timestamp', computing_window: int = None,
                 keep_all_timestamp: bool = False, fill_method: str = 'ffill', effective_number: int = None,
                 transformer: Transformer = ZenTransformer(), accumulator: Accumulator = None,
                 need_persist: bool = False, dry_run: bool = False, adjust_type: Union[AdjustType, str] = None) -> None:
        super().__init__(entity_schema, provider, entity_provider, entity_ids, exchanges, codes, the_timestamp,
                         start_timestamp, end_timestamp, columns, filters, order, limit, level, category_field,
                         time_field, computing_window, keep_all_timestamp, fill_method, effective_number, transformer,
                         accumulator, need_persist, dry_run, adjust_type)


if __name__ == '__main__':
    from zvt.drawer.drawer import Drawer

    df = get_kdata(entity_ids=['stock_sz_000338'], columns=['entity_id', 'timestamp', 'high', 'low', 'open', 'close'],
                   index=['entity_id', 'timestamp'], end_timestamp='2008-06-01')
    one_df = df[['high', 'low']]
    t = ZenTransformer()
    one_df = t.transform_one(one_df)

    print(one_df)

    #     annotation_df format:
    #                                     value    flag    color
    #     entity_id    timestamp

    ding = one_df[one_df.ding][['timestamp', 'high']]
    di = one_df[one_df.di][['timestamp', 'low']]

    df1 = ding.rename(columns={"high": "value"})
    df1['flag'] = '顶分型'

    df2 = di.rename(columns={"low": "value"})
    df2['flag'] = '底分型'

    factor_df: pd.DataFrame = pd.concat([df1, df2])
    factor_df = factor_df.sort_values(by=['timestamp'])
    factor_df['entity_id'] = 'stock_sz_000338'
    factor_df = factor_df.set_index(['entity_id', 'timestamp'])

    drawer = Drawer(main_df=df, annotation_df=factor_df)
    drawer.draw_kline(show=True)
