cd /data/Multitenanthakes
python3 controller/baseline/HQI/build_tree.py --min-size 10000
python3 controller/baseline/HQI/persist_tree.py --workers 8

cd /data/Multitenanthakes/basic_benchmark
python3 test_all.py --algorithm QDTree --efs 10
