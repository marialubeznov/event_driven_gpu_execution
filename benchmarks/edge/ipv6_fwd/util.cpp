#include "util.h"

uint32_t fastrand(uint64_t* seed)
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

void set_mac(uint8_t *mac_ptr, ULL mac_addr)
{
    mac_ptr[0] = mac_addr & 0xFF;
    mac_ptr[1] = (mac_addr >> 8) & 0xFF;
    mac_ptr[2] = (mac_addr >> 16) & 0xFF;
    mac_ptr[3] = (mac_addr >> 24) & 0xFF;
    mac_ptr[4] = (mac_addr >> 32) & 0xFF;
    mac_ptr[5] = (mac_addr >> 40) & 0xFF;
}

void print_mac_arr(uint8_t *mac)
{
    printf("%02X:%02X:%02X:%02X:%02X:%02X\n",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}


void print_ip_addr(uint32_t addr)
{
    printf("%d.%d.%d.%d\n",
        (addr >> 24 & 0xFF),
        (addr >> 16 & 0xFF),
        (addr >> 8 & 0xFF),
        (addr & 0xFF));
}
