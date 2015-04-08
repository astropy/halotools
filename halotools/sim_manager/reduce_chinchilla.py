import reader_tools
import pickle
from time import time

fname='hlist_1.00000.list'

column_info = ([
	(1, 'id', 'int'), 
	(5, 'pid', 'int'), 
	(6, 'upid', 'int'), 
	(10, 'mvir', 'float'), 
	(11, 'rvir', 'float'), 
	(12, 'rs', 'float'), 
	(15, 'scale_of_last_MM', 'float'), 
	(16, 'vmax', 'float'), 
	(17, 'x', 'float'), 
	(18, 'y', 'float'), 
	(19, 'z', 'float'), 
	(20, 'vx', 'float'), 
	(21, 'vy', 'float'), 
	(22, 'vz', 'float'), 
	(56, 'macc', 'float'), 
	(57, 'mpeak', 'float'), 
	(58, 'vacc', 'float'), 
	(59, 'vpeak', 'float'), 
	(60, 'halfmpeak_scale', 'float'), 
	(63, 'mar_tdyn', 'float'), 
	(66, 'mpeak_scale', 'float'), 
	(67, 'acc_scale', 'float'), 
	(68, 'm04_scale', 'float')
	])

mp_bolshoi = 1.35e8
mp_chinchilla = mp_bolshoi/8.
mpeak_cut = mp_chinchilla*100.
cuts = [(57, mpeak_cut, None)]

print ("Starting halo catalog reduction")
start = time()
halos=reader_tools.read_halocat(fname, column_info, input_halo_cuts=cuts)
end = time()
runtime = end - start
pickle.dump(halos, open("chinchilla_z0_halos.pickle", "wb"))

print("\n")
print("Total runtime = %.2f" % runtime)
print("\n")


"""
#scale(0) id(1) desc_scale(2) desc_id(3) num_prog(4) pid(5) upid(6) desc_pid(7) phantom(8) sam_mvir(9) mvir(10) rvir(11)
 rs(12) vrms(13) mmp?(14) scale_of_last_MM(15) vmax(16) x(17) y(18) z(19) vx(20) vy(21) vz(22) Jx(23) Jy(24) Jz(25) Spin
(26) Breadth_first_ID(27) Depth_first_ID(28) Tree_root_ID(29) Orig_halo_ID(30) Snap_num(31) Next_coprogenitor_depthfirst
_ID(32) Last_progenitor_depthfirst_ID(33) Rs_Klypin(34) Mvir_all(35) M200b(36) M200c(37) M500c(38) M2500c(39) Xoff(40) V
off(41) Spin_Bullock(42) b_to_a(43) c_to_a(44) A[x](45) A[y](46) A[z](47) b_to_a(500c)(48) c_to_a(500c)(49) A[x](500c)(5
0) A[y](500c)(51) A[z](500c)(52) T/|U|(53) M_pe_Behroozi(54) M_pe_Diemer(55) Macc(56) Mpeak(57) Vacc(58) Vpeak(59) Halfm
ass_Scale(60) Acc_Rate_Inst(61) Acc_Rate_100Myr(62) Acc_Rate_1*Tdyn(63) Acc_Rate_2*Tdyn(64) Acc_Rate_Mpeak(65) Mpeak_Sca
le(66) Acc_Scale(67) M4%_Scale(68)
"""











