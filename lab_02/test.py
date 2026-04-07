import math

R = 0.35
l = 12
L_k = 187 * (10**(-6))
C_k = 268 * (10**(-6))
R_k = 0.25
R_k2 = - 0.35
U_co = 1400
I_o = 3
T_w = 2000

print(U_co * math.sqrt(C_k / L_k))
print(2 * math.pi * math.sqrt(L_k * C_k))
