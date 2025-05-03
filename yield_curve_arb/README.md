# Economic Intuition Behind PCA-Based Yield Curve Trading

This strategy is based on the economic principle that yield curves tend to move in patterns
that can be decomposed into a few key factors (level, slope, curvature). Temporary deviations
from these patterns often mean-revert for several reasons:

1. **Central Bank Policy Transmission**: The ECB primarily controls short-term rates, while
   longer-term rates reflect growth and inflation expectations. This creates predictable
   relationships across the curve.

2. **Supply and Demand Imbalances**: Specific tenors can temporarily deviate due to
   supply/demand imbalances (pension fund buying, government issuance patterns, etc.)
   that later normalize.

3. **Market Segmentation**: Different investors focus on different parts of the curve,
   creating disconnects that eventually correct as arbitrageurs step in.

This model identifies these temporary dislocations and assumes they will mean-revert.



# Key Assumptions and Limitations

1. **Stationarity Assumption**: We assume that the PCA factors derived from our training window 
   are stable over time. During regime shifts (like the 2008 financial crisis or COVID-19), 
   these relationships can break down.

2. **Transaction Cost Simplification**: We don't model transaction costs realistically. 
   In practice, bid-ask spreads vary by tenor and market conditions.

3. **Holding Period Simplification**: We use a fixed holding period rather than dynamically 
   determining exit points based on convergence or stop-loss criteria.

4. **Financing Cost Omission**: We don't account for the cost of financing positions, which would 
   reduce real-world returns, especially for highly leveraged positions.

5. **Duration/DV01 Mismatch**: We treat each tenor point equally, but in reality, longer-dated 
   bonds have higher duration and thus higher price sensitivity to yield changes.



