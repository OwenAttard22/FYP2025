import math
        
def get_rt(simt):
    return -(math.exp(simt**2) - 1)

def get_rf(brg_to_wpt):
    return ((-8 * (brg_to_wpt-0.5)**2) + 1)

def get_rd(dist_to_wpt):
    if dist_to_wpt < 0.5:
        return (1 - (2 * dist_to_wpt)**0.5)
    else:
        return 0
    
def get_rn(n1_dist, n2_dist):
    r_n1 = -((1-n1_dist)**2) if n1_dist <= 1 else 0
    r_n2 = -((1-n2_dist)**2) if n2_dist <= 1 else 0
    return r_n1 + r_n2
    
test = [0,0.5,1]
for i in test:
    print(f"get_rt({i}) = {get_rt(i)}")
print("-" * 40)

for i in test:
    print(f"get_rf({i}) = {get_rf(i)}")
print("-" * 40)
    
for i in test:
    print(f"get_rd({i}) = {get_rd(i)}")
print("-" * 40)
    
for i in test:
    print(f"get_rn({i}) = {get_rn(i, i)}")
print("-" * 40)