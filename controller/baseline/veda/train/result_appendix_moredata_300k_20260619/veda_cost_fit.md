# Veda Appendix B Cost Model Fit

Formula:

`C_theta(N,efs)=a*log2(1+N)+b*efs+c`

latency_stat = `median`
a = 0.0388485829
b = 0.0237531138
c = 0.8879201454

size_sweep_r2 = 0.503869
efs_linear_r2 = 0.884145
efs_log_r2 = 0.842245
selected_efs_term_by_r2 = `linear_efs`

fixed_size = 300000
sizes = 1000,3000,5000,10000,20000,50000,80000,100000,200000,300000
efs_values = 1,5,10,20,40,80,120,200,400,800
query_count = 500
