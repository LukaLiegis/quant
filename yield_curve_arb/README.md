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