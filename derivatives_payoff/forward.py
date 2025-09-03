import numpy as np
import matplotlib.pyplot as plt

K = 100
S_min, S_max = 50, 150
S_T = np.linspace(S_min, S_max, K)

long_payoff = S_T - K
short_payoff = K - S_T

plt.figure(figsize=(12, 8))
plt.plot(S_T, long_payoff, 'b-',label="Long Forward")
plt.plot(S_T, short_payoff, 'r-', label="Short Forward")
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=K, color='k', linestyle='--')

plt.title("Forward Contract Payoff")
plt.xlabel("Underlying Price")
plt.ylabel("Payoff")
plt.grid(True)
plt.legend()
plt.savefig("derivatives_payoff/forward_payoff.png")
plt.show()