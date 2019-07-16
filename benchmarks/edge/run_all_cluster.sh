#!/bin/bash
cd ipv4_fwd
./run_all_cluster.sh
cd ../ipv6_fwd
./run_all_cluster.sh
cd ../memc_conv
./run_all_cluster.sh
