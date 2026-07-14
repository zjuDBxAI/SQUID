# Veda Appendix B Cost Model Fit

Formula:

`C_theta(N,efs)=a*log2(1+N)+b*efs+c`

latency_stat = `median`
a = 0.0490324038
b = 0.0406161915
c = 0.0000000000

joint_fit_r2 = 0.975763
size_sweep_r2 = 0.634836
efs_linear_r2 = 0.971743
efs_log_r2 = 0.968107
selected_efs_term_by_r2 = `linear_efs`

fixed_size = 480000
sizes = 10000,50000,100000,300000,480000
efs_values = 1,5,10,20,40,80,120,200,400,800,1000
query_count = 50
