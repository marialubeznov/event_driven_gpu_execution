#!/bin/bash
cd ipv4_fwd
./run_10G.sh
./run_5G.sh
./run_1G.sh
cd ../ipv6_fwd
./run_1G.sh
./run_5G.sh
./run_10G.sh
cd ../memc_conv
./run_1G.sh
./run_5G.sh
./run_10G.sh
