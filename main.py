from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List
import jsonpickle
from typing import Any
import json
import numpy as np
import collections

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


logger = Logger()


def fillBids(price, amount, product, state) -> [int, List[Order]]:
    orders = []
    buy_orders = state.order_depths[product].buy_orders
    buy_orders = dict(sorted(buy_orders.items(), reverse=True))

    try:
        position = state.position[product]
    except KeyError:
        position = 0
    position_limit_positive = 20

    for buy_price, buy_amount in buy_orders.items():
        if buy_price > price:
            available_amount = max(min(amount, position_limit_positive + position), 0)
            if available_amount != 0:
                buy_orders[buy_price] -= buy_amount
                orders = [(Order(product, buy_price, -buy_amount))]
                position -= abs(buy_amount)

    return position, orders


def fillAsks(price, amount, product, state) -> [int, List[Order]]:
    orders = []
    sell_orders = state.order_depths[product].sell_orders
    sell_orders = dict(sorted(sell_orders.items()))

    try:
        position = state.position[product]
    except KeyError:
        position = 0
    position_limit_positive = 20

    for sell_price, sell_amount in sell_orders.items():
        if sell_price < price:
            available_amount = - max(min(amount, position_limit_positive - position), 0)
            if available_amount != 0:
                sell_orders[sell_price] -= sell_amount
                orders = [(Order(product, sell_price, -sell_amount))]
                position += abs(sell_amount)

    return position, orders


def values_extract(order_dict, buy=0):
    tot_vol = 0
    best_val = -1
    mxvol = -1

    for ask, vol in order_dict.items():
        if (buy == 0):
            vol *= -1
        tot_vol += vol
        if tot_vol > mxvol:
            mxvol = vol
            best_val = ask

    return tot_vol, best_val


def process_STARFRUIT(state: TradingState, price) -> List[Order]:
    orders: list[Order] = []
    order_depth = state.order_depths["STARFRUIT"]
    osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
    obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

    sell_vol, best_sell_pr = values_extract(osell)
    buy_vol, best_buy_pr = values_extract(obuy, 1)
    product = "STARFRUIT"
    LIMIT = 20
    cpos = state.position.get("STARFRUIT", 0)


    nxt_price = get_weighted_average(state, "STARFRUIT")
    acc_bid = int(round(nxt_price)) - 1
    acc_ask = int(round(nxt_price)) + 1

    for ask, vol in osell.items():
        if ((ask <= acc_bid) or ((state.position.get("STARFRUIT", 0) < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
            order_for = min(-vol, LIMIT - cpos)
            cpos += order_for
            assert (order_for >= 0)
            orders.append(Order(product, ask, order_for))

    undercut_buy = best_buy_pr + 1
    undercut_sell = best_sell_pr - 1

    bid_pr = min(undercut_buy, acc_bid)  # we will shift this by 1 to beat this price
    sell_pr = max(undercut_sell, acc_ask)

    if cpos < LIMIT:
        num = LIMIT - cpos
        orders.append(Order(product, bid_pr, num))
        cpos += num

    cpos = state.position.get("STARFRUIT", 0)

    for bid, vol in obuy.items():
        if ((bid >= acc_ask) or ((state.position.get("STARFRUIT", 0) > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
            order_for = max(-vol, -LIMIT - cpos)
            # order_for is a negative number denoting how much we will sell
            cpos += order_for
            assert (order_for <= 0)
            orders.append(Order(product, bid, order_for))

    if cpos > -LIMIT:
        num = -LIMIT - cpos
        orders.append(Order(product, sell_pr, num))
        cpos += num

    return orders


def process_AMETHYSTS(state: TradingState) -> List[Order]:
    product = "AMETHYSTS"
    orders: list[Order] = []
    order_depth = state.order_depths[product]

    osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
    obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

    sell_vol, best_sell_pr = values_extract(osell)
    buy_vol, best_buy_pr = values_extract(obuy, 1)

    cpos = state.position.get("AMETHYSTS", 0)
    acc_bid = 10000
    acc_ask = 10000
    POSITION_LIMIT = 20
    mx_with_buy = -1

    for ask, vol in osell.items():
        if ((ask < acc_bid) or ((state.position.get("AMETHYSTS", 0) < 0) and (ask == acc_bid))) and cpos < POSITION_LIMIT:
            mx_with_buy = max(mx_with_buy, ask)
            order_for = min(-vol, POSITION_LIMIT - cpos)
            cpos += order_for
            assert (order_for >= 0)
            orders.append(Order(product, ask, order_for))


    undercut_buy = best_buy_pr + 1
    undercut_sell = best_sell_pr - 1

    bid_pr = min(undercut_buy, acc_bid - 1)  # we will shift this by 1 to beat this price
    sell_pr = max(undercut_sell, acc_ask + 1)

    if (cpos < POSITION_LIMIT) and (state.position.get("AMETHYSTS", 0)< 0):
        num = min(40, POSITION_LIMIT - cpos)
        orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
        cpos += num

    if (cpos < POSITION_LIMIT) and (state.position.get("AMETHYSTS", 0) > 15):
        num = min(40, POSITION_LIMIT - cpos)
        orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
        cpos += num

    if cpos < POSITION_LIMIT:
        num = min(40, POSITION_LIMIT - cpos)
        orders.append(Order(product, bid_pr, num))
        cpos += num

    cpos = state.position.get("AMETHYSTS", 0)

    for bid, vol in obuy.items():
        if ((bid > acc_ask) or ((state.position.get("AMETHYSTS", 0) > 0) and (bid == acc_ask))) and cpos > -POSITION_LIMIT:
            order_for = max(-vol, -POSITION_LIMIT - cpos)
            # order_for is a negative number denoting how much we will sell
            cpos += order_for
            assert (order_for <= 0)
            orders.append(Order(product, bid, order_for))

    if (cpos > -POSITION_LIMIT) and (state.position.get("AMETHYSTS", 0) > 0):
        num = max(-40, -POSITION_LIMIT - cpos)
        orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
        cpos += num

    if (cpos > -POSITION_LIMIT) and (state.position.get("AMETHYSTS", 0) < -15):
        num = max(-40, -POSITION_LIMIT - cpos)
        orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
        cpos += num

    if cpos > -POSITION_LIMIT:
        num = max(-40, -POSITION_LIMIT - cpos)
        orders.append(Order(product, sell_pr, num))
        cpos += num

    return orders


def get_weighted_average(state: TradingState, product: str) -> float:
    # calculate the weighted average of the starfruit price

    total_bid_price = 0
    total_bid_amount = 0
    total_ask_price = 0
    total_ask_amount = 0

    for price, amount in state.order_depths[product].buy_orders.items():
        total_bid_price += price * abs(amount)
        total_bid_amount += abs(amount)

    for price, amount in state.order_depths[product].sell_orders.items():
        total_ask_price += price * abs(amount)
        total_ask_amount += abs(amount)

    weighted_average_bid = total_bid_price / total_bid_amount
    weighted_average_ask = total_ask_price / total_ask_amount

    weighted_average = (weighted_average_ask + weighted_average_bid) / 2
    return weighted_average


def compute_fair_buy_price(import_tariff, conversion_ask, transport_fees):
    return import_tariff + conversion_ask + transport_fees


def compute_fair_sell_price(export_tariff, conversion_bid, transport_fees):
    return conversion_bid - export_tariff - transport_fees


def get_mid_price(state: TradingState, product: str):
    # calculate the mid price by the best bid and best ask
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    sell_orders = order_depth.sell_orders

    # if there isnt any buy or sell orders, return nan
    if not buy_orders or not sell_orders:
        return None

    best_buy = max(buy_orders.keys())
    best_sell = min(sell_orders.keys())

    mid_price = (best_buy + best_sell) / 2
    return mid_price


def get_true_mid_price(state: TradingState, product: str) -> float:
    weighted_mid = get_weighted_average(state, product)
    order_depth = state.order_depths[product]
    buy_orders = order_depth.buy_orders
    sell_orders = order_depth.sell_orders

    # remove all buy and sell orders whose price falls within +- 2 of the weighted average
    buy_orders = {price: amount for price, amount in buy_orders.items() if price < weighted_mid - 2 or price > weighted_mid + 2}
    sell_orders = {price: amount for price, amount in sell_orders.items() if price < weighted_mid - 2 or price > weighted_mid + 2}

    best_buy = max(buy_orders.keys())
    best_sell = min(sell_orders.keys())

    mid_price = (best_buy + best_sell) / 2

    return mid_price


def tradeBasket(state):
    orders = {}

    max_position = 60

    product = "GIFT_BASKET"

    c_pos = state.position.get(product, 0)

    chocolate = 4
    strawberry = 6
    rose = 1

    chocolate_p = get_mid_price(state, "CHOCOLATE")
    strawberry_p = get_mid_price(state, "STRAWBERRIES")
    rose_p = get_mid_price(state, "ROSES")
    gift_basket_p = get_mid_price(state, "GIFT_BASKET")

    combined_price = chocolate_p * chocolate + strawberry_p * strawberry + rose_p * rose
    diff = gift_basket_p - combined_price - 379.49
    std = 69.389


    best_bid = max(state.order_depths["GIFT_BASKET"].buy_orders.keys())
    best_ask = min(state.order_depths["GIFT_BASKET"].sell_orders.keys())

    percent = 0.55
    offset = 2

    if diff > std * percent:
        amt = c_pos + max_position
        orders["GIFT_BASKET"] = [Order("GIFT_BASKET", best_bid - offset, -amt)]
    elif diff < -std * percent:
        amt = max_position - c_pos
        orders["GIFT_BASKET"] = [Order("GIFT_BASKET", best_ask + offset, amt)]


    return orders.get("GIFT_BASKET",[])


def tradeOrchid(state):
    orders = {}
    orders = []
    product = "ORCHIDS"
    order_depth = state.order_depths["ORCHIDS"]
    lowest_sell_offer = min(order_depth.sell_orders.items())[0]
    highest_buy_order = max(order_depth.buy_orders.items())[0]

    obs = state.observations.conversionObservations[product]
    position_limit = 100
    old_pos = state.position.get(product, 0)
    max_pos = old_pos  # after all buys
    min_pos = old_pos  # after all sells
    instant_buy = obs.askPrice + obs.transportFees + obs.importTariff  # sell offer from south
    instant_sell = obs.bidPrice - obs.transportFees - obs.exportTariff  # buy order from south
    logger.print("PRICE TO INSTANTLY BUY FROM SOUTH: " + str(instant_buy))
    logger.print("COST TO INSTANTLY SELL TO SOUTH: " + str(instant_sell))

    # arbitrage market taking
    if len(order_depth.sell_orders) > 1:
        for ask, volume in list(order_depth.sell_orders.items()):
            if ask < instant_sell:
                volume = -max(volume, max_pos - position_limit)
                logger.print("+autobuy", str(volume), "for", ask)
                orders.append(Order(product, ask, volume))
                max_pos += volume
            else:
                lowest_sell_offer = min(lowest_sell_offer, ask)
    if len(order_depth.buy_orders) > 1:
        for ask, volume in list(order_depth.buy_orders.items()):
            if ask > instant_buy:
                volume = -min(volume, min_pos + position_limit)
                logger.print("+autosell", str(volume), "for", ask)
                orders.append(Order(product, ask, volume))
                min_pos += volume
            else:
                highest_buy_order = max(highest_buy_order, ask)

    # arbitrage market making!
    mid = (lowest_sell_offer + highest_buy_order) / 2.0
    true_mid = get_true_mid_price(state, "ORCHIDS")

    margins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3]  # [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3]
    fill_chance = [0.0953909834, 0.1781911754, 0.5746515892, 0.9493771493, 0.9783989235, 0.9787609087,
                   1]  # [0.0319905213, 0.0746445498, 0.2109004739, 0.2748815166, 0.5829383886, 0.6954976303, 0.7132701422, 0.7132701422, 0.7132701422, 0.778436019, 0.8637440758, 1]

    # instant_buy  #sell offer from south
    # instant_sell #buy order from south

    # best profit for buy orders
    best_export_margin = -1
    best_export_profit = -1
    for i in range(0, 7):
        maybe_price = round(true_mid + margins[i])
        if maybe_price == (true_mid + margins[i]):  # check to see if margin is correct
            instant_sell_profit = instant_sell - maybe_price
            projected_profit = instant_sell_profit * fill_chance[i]
            # logger.print("Predicted buy order profit for margin "+str(margins[i])+" is "+str(projected_profit))
            if projected_profit > best_export_profit:
                best_export_margin = margins[i]
                best_export_profit = projected_profit
    logger.print("Predicted best buy order profit for margin " + str(best_export_margin) + " is " + str(
        best_export_profit))

    if best_export_profit > 0 and best_export_margin != -1:  # REMEMBER TO ENABLE
        # set up buy order at mid + best_margin
        buy_price = int(round(true_mid + best_export_margin))
        volume = min(-max_pos + position_limit, 100)
        logger.print("+insane arb buy of ", str(volume), "for", buy_price)
        orders.append(Order(product, buy_price, volume))
        max_pos += volume
        orchidState = "ARB"

    else:
        # weak arb
        buy_price = int(round(instant_sell - 3.14159265358979 / 1.618033988749))
        volume = min(-max_pos + position_limit, 100)
        logger.print("+weak arb buy of ", str(volume), "for", buy_price)
        orders.append(Order(product, buy_price, volume))
        max_pos += volume
        orchidState = "ARB"

    # best profit for sell offers
    best_import_margin = -1
    best_import_profit = -1
    for i in range(0, 7):
        maybe_price = round(true_mid - margins[i])
        if maybe_price == (true_mid - margins[i]):  # check to see if margin is correct
            instant_buy_profit = maybe_price - instant_buy
            projected_profit = instant_buy_profit * fill_chance[i]
            # logger.print("Predicted sell order profit for margin "+str(margins[i])+" is "+str(projected_profit))
            if projected_profit > best_import_profit:
                best_import_margin = margins[i]
                best_import_profit = projected_profit
    logger.print("Predicted best sell order profit for margin " + str(best_import_margin) + " is " + str(
        best_import_profit))

    if best_import_profit > 0 and best_import_margin != -1:
        # set up sell order at mid - best_margin
        sell_price = int(round(true_mid - best_import_margin))
        volume = max(-min_pos - position_limit, -100)
        logger.print("+insane arb sell of ", str(volume), "for", sell_price)
        orders.append(Order(product, sell_price, volume))
        min_pos += volume
        orchidState = "ARB"

    else:
        # weak arb
        sell_price = int(round(instant_buy + 3.14159265358979 / 1.618033988749))
        volume = max(-min_pos - position_limit, -100)
        logger.print("+weak arb sell of ", str(volume), "for", sell_price)
        orders.append(Order(product, sell_price, volume))
        min_pos += volume
        orchidState = "ARB"

    return orders


def tradeCoconuts(state):
    orders = {}
    coupon_max_pos = 600
    coco_max_pos = 300

    product = "COCONUT_COUPON"
    product1 = "COCONUT"

    coupon_mid = get_mid_price(state, product)
    coco_mid = get_mid_price(state, product1)

    if not coupon_mid or not coco_mid:
        return {}

    coef_2 = 9364.904710

    intrinsic = coco_mid - 10000
    delta = 0.493423
    time_value = coupon_mid + (0.5) * intrinsic

    time_value = time_value + coef_2

    diff = coco_mid - time_value

    best_bid_coupon = max(state.order_depths[product].buy_orders.keys())
    best_ask_coupon = min(state.order_depths[product].sell_orders.keys())

    best_bid_coco = max(state.order_depths[product1].buy_orders.keys())
    best_ask_coco = min(state.order_depths[product1].sell_orders.keys())

    # perform pair trading between COCONUT_COUPON and COCONUT where coupon will rise if diff is negative and vice versa

    cpos_coupon = state.position.get(product, 0)
    cpos_coco = state.position.get(product1, 0)
    # Calculate the percentage based on the magnitude of diff

    percentage = 1
    std = 13.38
    threshold = std * 0.5

    if diff > threshold:
        # Adjust coupon_max_pos based on the percentage
        adjusted_coupon_max_pos = int(coupon_max_pos * percentage)
        amt_coupon = adjusted_coupon_max_pos - cpos_coupon
        amt_coco = coco_max_pos + cpos_coco
        orders[product] = [Order(product, best_bid_coupon + 1, amt_coupon)]
        orders[product1] = [Order(product1, best_ask_coco - 1, -amt_coco)]

    if diff < -threshold:
        # Adjust coupon_max_pos based on the percentage
        adjusted_coupon_max_pos = int(coupon_max_pos * percentage)
        amt_coupon = adjusted_coupon_max_pos + cpos_coupon
        amt_coco = coco_max_pos - cpos_coco
        orders[product] = [Order(product, best_ask_coupon - 1, -amt_coupon)]
        orders[product1] = [Order(product1, best_bid_coco + 1, amt_coco)]

    return orders


def tradeRoses(state):
    orders = {}
    market_trades = state.market_trades
    max_rose = 60
    # get roses trades by rhianna
    rose_trades = market_trades.get("ROSES", [])
    for trade in rose_trades:
        if trade.seller == "Rhianna":
            cpos = state.position.get("ROSES", 0)
            amt = max_rose + cpos
            orders["ROSES"] = [Order("ROSES", trade.price - 10, -amt)]
        if trade.buyer == "Rhianna":
            cpos = state.position.get("ROSES", 0)
            amt = max_rose - cpos
            orders["ROSES"] = [Order("ROSES", trade.price + 10, amt)]
    return orders.get("ROSES", [])

class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        orchid_pos = state.position.get("ORCHIDS", 0)
        conversions = -orchid_pos


        orders["GIFT_BASKET"] = tradeBasket(state)
        orders["ORCHIDS"] = tradeOrchid(state)
        orders["ROSES"] = tradeRoses(state)
        orders["AMETHYSTS"] = process_AMETHYSTS(state)
        orders["STARFRUIT"] = process_STARFRUIT(state, price=0)
        orders["COCONUT_COUPON"] = tradeCoconuts(state).get("COCONUT_COUPON", [])
        orders["COCONUT"] = tradeCoconuts(state).get("COCONUT", [])



        logger.flush(state, orders, conversions, trader_data)
        # dcobj = [state, orders, conversions]
        # print(jsonpickle.encode(dcobj))
        return orders, conversions, trader_data
