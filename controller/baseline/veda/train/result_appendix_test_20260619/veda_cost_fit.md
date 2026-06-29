# Veda Appendix B Cost Model Fit

Formula:

`C_theta(N,efs)=a*log2(1+N)+b*efs+c`

latency_stat = `median`
a = 0.0450317113
b = 0.0168740079
c = 0.3060789741

size_sweep_r2 = 0.571445
efs_linear_r2 = 0.988366
efs_log_r2 = 0.974761
selected_efs_term_by_r2 = `linear_efs`

fixed_size = 300000
sizes = 1000,3000,10000,30000,100000,300000,600000,1000000
efs_values = 1,5,10,20,40,80,120,200,400,800
query_count = 200
