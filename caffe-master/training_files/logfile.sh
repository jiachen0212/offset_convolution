LOG=./examples/cifar10/log/svd_energy_0.4_ts_BN-log-`date +%Y-%m-%d-%H-%M-%S`.log
./build/tools/caffe train --solver=./examples/cifar10/VGG9/solver.prototxt -gpu=6,7 2>&1 | tee $LOG