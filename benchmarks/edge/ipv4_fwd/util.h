#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <net/ethernet.h>

#define LL long long
#define ULL unsigned long long

void set_mac(uint8_t *mac_ptr, ULL mac_addr);
uint32_t fastrand(uint64_t* seed);
void print_mac_arr(uint8_t* mac);
void print_ip_addr(uint32_t addr);
