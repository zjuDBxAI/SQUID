# Veda Appendix B Cost Model Fit - Filtered Monotonic

Filter: dropped size-sweep `N=480000, efs=1` because median latency was lower than `N=300000`.

Formula:

`C_theta(N,efs)=a*log2(1+N)+b*efs+c`

a = 0.0495519449
b = 0.0406021420
c = 0.0000000000

## Stage 1 Size Sweep

a_size = 0.0513403570
c1 = 0.0000000000
R2 = 0.672442
RMSE = 0.118910 ms
MAE = 0.109812 ms

## Stage 2 EFS Sweep

b_linear = 0.0405942528
c2_linear = 0.9407072347
linear_R2 = 0.971743
linear_RMSE = 2.297712 ms
linear_MAE = 1.535008 ms

b_log = 0.0040678536
c2_log = 1.8125764495
log_R2 = 0.968107

## Joint Fit

joint_R2 = 0.975255
joint_RMSE = 1.968685 ms
joint_MAE = 1.155037 ms

