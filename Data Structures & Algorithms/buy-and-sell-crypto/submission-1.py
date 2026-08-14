class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        best_profit = 0
        best_buy = 100
        for i in range(len(prices)):
            best_profit = max(best_profit, prices[i] - best_buy)
            best_buy = min(prices[i], best_buy)
        return best_profit