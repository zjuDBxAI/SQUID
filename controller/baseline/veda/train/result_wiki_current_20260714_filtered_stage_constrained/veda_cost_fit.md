# Veda Appendix B Cost Model Fit - Stage Constrained, Filtered Size Sweep

Formula:

`C_theta(N,efs)=a*log2(1+N)+b*efs+c`

latency_stat = `median`
a = 0.0513403570
b = 0.0405942528
c = 0.0000000000
c_from_size_raw = -0.0405942528
c_from_efs_raw = -0.0282227862
c_from_size_constrained = 0.0000000000
c_from_efs_constrained = 0.0000000000

joint_fit_r2 = 0.975255
size_sweep_r2 = 0.672442
efs_linear_r2 = 0.971743
efs_log_r2 = 0.968107
selected_efs_term_by_r2 = `linear_efs`

fixed_size = 480000
sizes = 10000,50000,100000,300000
efs_values = 1,5,10,20,40,80,120,200,400,800,1000
