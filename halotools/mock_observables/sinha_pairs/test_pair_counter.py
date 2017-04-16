import wrapper
import numpy as np 

Lbox = 1000.0 # Size of imaginary simulation

# Create a tightly localized box of points
Npts1 = 5.e3
mini_Lbox = 1
box1_center = 5
box1_min = box1_center - 0.5*mini_Lbox
box1_max = box1_center + 0.5*mini_Lbox
d1 = np.random.uniform(box1_min, box1_max, Npts1*3).reshape(Npts1, 3)

# Create another tightly localized box at a distant location
Npts2 = 1.5e3
mini_box_separations = 100 
box2_center = box1_center + mini_box_separations
box2_min = box2_center - 0.5*mini_Lbox
box2_max = box2_center + 0.5*mini_Lbox
d2 = np.random.uniform(box2_min, box2_max, Npts2*3).reshape(Npts2, 3)


rbins = [0, 5, 10, 25, 200]

# Now let's test the correctness of our cross-pair counter
# Since we've set up our bins so that *all* (d1, d2) pairs 
# will land in the outermost bin, then we know what the results 
# of the pair counter should be analytically, a non-trivial test 
# of the correctness of both the wrapper and the pair counter. 
correct_outermost_result = int(Npts1*Npts2)
print("cross-correlation results:")
wrapper.cy_countpairs(Lbox, rbins, d1, d2)
print("Result in outermost bin should be %i ") % correct_outermost_result
# Looks good. 

###

# Now let's check the auto-correlation branch

# Concatenate our two sets of points into a joint sample
Npts_total = Npts1 + Npts2
d1d2 = np.append(d1, d2).reshape(Npts_total, 3)

# For any set of Npts points, 
# the total number of unique pairs of those points is Npts*(Npts-1)/2, 
# the number of non-zero entries in an upper triangular matrix.
# We have set up our binning scheme so that the innermost bin 
# will contain *all* (d1, d1) pairs, *and* all (d2, d2) pairs. 
# This again gives us an analytically known result to compare to.

correct_innermost_result = int((Npts1*(Npts1-1)/2.) + (Npts2*(Npts2-1)/2.))

print("")
print("auto-correlation:")
wrapper.cy_countpairs(Lbox, rbins, d1d2)

print("Result in innermost bin should be %i ") % correct_innermost_result

correct_outermost_result = int(Npts1*Npts2)
print("Result in outermost bin should be %i ") % correct_outermost_result

# Ok, those auto-pair count results were wrong. Here's what I think is going wrong. 
# It looks like the auto-pair counter is double-counting pairs. If pairs are being double-counted, 
# the results would be the following:
double_counted_innermost_result = int(Npts1*Npts1 + Npts2*Npts2)
double_counted_outermost_result = int(2.*Npts1*Npts2)
print("Incorrectly double-counted result in innermost bin would be %i ") % double_counted_innermost_result
print("Incorrectly double-counted result in outermost bin would be %i ") % double_counted_outermost_result

# To my eye, your wrapper appears to be correctly written. 
# In particular, the control flow around the autocorr int looks good to me. 
# So this must mean that either I have mis-interpreted the meaning of this variable, 
# or there is a bug in Manodeep's code. 





