#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "ipv6.h"
#include "util.h"


#define IPv6_ISSET(a, i) (a & (1 << i))
#define IPv6_MAX_ETHPORTS 16

/**< Count the number of 1-bits in n */
int ipv6_bitcount(int n)
{
	int count = 0;
	while(n > 0) {
		count ++;
		n = n & (n - 1);
	}
	return count;
}

/**< Returns an array containing the port numbers of all active ports */
int *ipv6_get_active_ports(int portmask)
{
	int num_active_ports = ipv6_bitcount(portmask);
	int *active_ports = (int *) malloc(num_active_ports * sizeof(int));
	int pos = 0, i;
	for(i = 0; i < IPv6_MAX_ETHPORTS; i++) {
		if(IPv6_ISSET(portmask, i)) {
			active_ports[pos] = i;
			pos ++;
		}
	}
	assert(pos == num_active_ports);
	return active_ports;
}




/**< Generate an LPM6 struct with prefixes mapped to ports from portmask.
  *
  *  o The server calls this function with add_prefixes = 1 to get a
  *  populated lpm6 struct.
  * 
  *  o A client calls with add_prefixes = 0 to get the random prefixes. These
  *  can be later extended to generate the probe IPv6 addresses.
  * 
  *  o ipv6_gen_rand_prefixes() seeds rand() so same prefixes are used always.
  */
struct rte_lpm6 *ipv6_init(int portmask,
    struct ipv6_prefix **prefix_arr, int add_prefixes)
{
    int i;

    /**< Get random prefixes */
    int num_prefixes = IPV6_NUM_RAND_PREFIXES;
    *prefix_arr = ipv6_gen_rand_prefixes(num_prefixes, portmask);

    if(add_prefixes == 1) {
        struct rte_lpm6_config ipv6_config;
        ipv6_config.max_rules = 1000000;
        ipv6_config.number_tbl8s = IPV6_NUM_TBL8;
        assert(num_prefixes < (int) ipv6_config.max_rules);

        struct rte_lpm6 *lpm = rte_lpm6_create(0, &ipv6_config);

        for(i = 0; i < num_prefixes; i ++) {
            /**< Add this prefix to LPM6 */
            struct ipv6_prefix cur_prfx = (*prefix_arr)[i];

            int add_status = rte_lpm6_add(lpm,
                cur_prfx.bytes, cur_prfx.depth, cur_prfx.dst_port);

            if(add_status < 0) {
                printf("ipv6: Failed to add IPv6 prefix %d. Status = %d\n",
                    i, add_status);
                exit(-1);
            }

            if(i % 1000 == 0) {
                printf("ipv6: Added prefixes = %d, total = %d\n",
                    i, num_prefixes);
            }
        }

        printf("\tipv6: Done inserting prefixes\n");
        return lpm;
    } else {
        return NULL;
    }
}

/**< Read IPv6 prefixes from a file */
struct ipv6_prefix *ipv6_read_prefixes(const char *prefixes_file,
    int *num_prefixes)
{
    assert(prefixes_file != NULL && num_prefixes != NULL);

    FILE *prefix_fp = fopen(prefixes_file, "r");
    assert(prefix_fp != NULL);

    fscanf(prefix_fp, "%d", num_prefixes);
    assert(*num_prefixes > 0);
    printf("ipv6: Reading %d prefixes\n", *num_prefixes);

    int prefix_mem_size = *num_prefixes * sizeof(struct ipv6_prefix);
    struct ipv6_prefix *prefix_arr = (struct ipv6_prefix*)malloc(prefix_mem_size);
    assert(prefix_arr != NULL);

    int i, j;
    for(i = 0; i < *num_prefixes; i ++) {
        /**< A prefix is formatted as <depth> <bytes 0 ... 15> <dst port> */
        fscanf(prefix_fp, "%d", &prefix_arr[i].depth);

        for(j = 0; j < IPV6_ADDR_LEN; j ++) {
            int new_byte;
            fscanf(prefix_fp, "%d", &new_byte);
            assert(new_byte >= 0 && new_byte <= 255);

            prefix_arr[i].bytes[j] = new_byte;
        }

        fscanf(prefix_fp, "%d", &prefix_arr[i].dst_port);
    }

    return prefix_arr;    
}

/**< Generate IPv6 prefixes with random content and depth b/w 48-64. The dst
  *  ports are chosen from portmask. */
struct ipv6_prefix *ipv6_gen_rand_prefixes(int num_prefixes, int portmask)
{
    assert(num_prefixes > 0);
    printf("ipv6: Generating %d random prefixes\n", num_prefixes);

    int num_active_ports = ipv6_bitcount(portmask);
    int *port_arr = ipv6_get_active_ports(portmask);
    
    /**< Seed rand() before generating prefixes so that server and clients
      *  generate the same set of prefixes. */
    srand(IPV6_RAND_PREFIXES_SEED);

    int prefix_mem_size = num_prefixes * sizeof(struct ipv6_prefix);
    struct ipv6_prefix *prefix_arr = (struct ipv6_prefix*)malloc(prefix_mem_size);
    assert(prefix_arr != NULL);

    int i, j;
    for(i = 0; i < num_prefixes; i ++) {

        /**< Prefix depth is between 48 and 64 */
        prefix_arr[i].depth = (rand() % 17) + 48;

        for(j = 0; j < IPV6_ADDR_LEN; j ++) {
            prefix_arr[i].bytes[j] = rand() % 256;
        }

        /**< Select the dst port from active ports */
        prefix_arr[i].dst_port = port_arr[rand() % num_active_ports];
    }

    return prefix_arr;
}

/**< Increase the number of prefixes in prefix_arr */
struct ipv6_prefix *ipv6_amp_prefixes(struct ipv6_prefix *prefix_arr,
    int num_prefixes, int amp_factor)
{
    int mem_size = num_prefixes * amp_factor * sizeof(struct ipv6_prefix);
    struct ipv6_prefix *new_prefix_arr = (struct ipv6_prefix*)malloc(mem_size);
    assert(new_prefix_arr != NULL);

    struct ipv6_perm *perm_arr = ipv6_gen_perms(amp_factor);

    int i, j, k;
    for(i = 0; i < num_prefixes * amp_factor; i += amp_factor) {

        /**< New prefixes i, ..., i + amp_factor - 1 come from old prefix
          *  numbered i / amp_factor */
        for(j = 0; j < amp_factor; j ++) {
            new_prefix_arr[i + j] = prefix_arr[i / amp_factor];

            /**< Transform only the valid bytes */
            int bytes_to_transform = prefix_arr[i / amp_factor].depth / 8;

            for(k = 0; k < bytes_to_transform; k ++) {
                int old_byte = new_prefix_arr[i + j].bytes[k];
                int new_byte = perm_arr[j].P[old_byte];

                new_prefix_arr[i + j].bytes[k] = new_byte;
            }
        }
    }

    return new_prefix_arr;
}

/**< Generate probe IPv6 addresses from prefixes */
struct ipv6_addr *ipv6_gen_addrs(int num_addrs,
    struct ipv6_prefix *prefix_arr, int num_prefixes)
{
    assert(num_addrs > 0 && prefix_arr != NULL && num_prefixes > 0);

    struct ipv6_addr *addr_arr;
    int addr_mem_size = num_addrs * sizeof(struct ipv6_addr);

    addr_arr = (struct ipv6_addr*)lpm6_hrd_malloc_socket(PROBE_ADDR_SHM_KEY, addr_mem_size, 0);

    /**< Generate addresses using randomly chosen prefixes */
    int i, j;
    uint64_t seed = 0xdeadbeef;

    for(i = 0; i < num_addrs; i ++) {
        int prefix_id = rand() % num_prefixes;
        int prefix_depth = prefix_arr[prefix_id].depth;
        int last_full_byte = (prefix_depth / 8) - 1;
        assert(last_full_byte >= 0 && last_full_byte < IPV6_ADDR_LEN);

        for(j = 0; j < IPV6_ADDR_LEN; j ++) {
            addr_arr[i].bytes[j] = prefix_arr[prefix_id].bytes[j];
        }

        for(j = last_full_byte + 1; j < IPV6_ADDR_LEN; j ++) {
            addr_arr[i].bytes[j] += fastrand(&seed) % 128;
            addr_arr[i].bytes[j] %= 256;
        }
    }

    return addr_arr;
}

void ipv6_print_prefix(struct ipv6_prefix *prefix)
{
    int i;
    printf("depth: %d, bytes: ", prefix->depth);
    for(i = 0; i < IPV6_ADDR_LEN; i ++) {
        printf("%d ", prefix->bytes[i]);
    }

    printf(" dst_port: %d\n", prefix->dst_port);
}

void ipv6_print_addr(struct ipv6_addr *addr)
{
    int i;
    for(i = 0; i < IPV6_ADDR_LEN; i ++) {
        printf("%d ", addr->bytes[i]);
    }

    printf("\n");
}

/**< Generate N different permutations of 0, ..., 255 */
struct ipv6_perm *ipv6_gen_perms(int N)
{
    struct ipv6_perm *res = (struct ipv6_perm*)malloc(N * sizeof(struct ipv6_perm));
    assert(res != 0);

    int i, j;
    for(i = 0; i < N; i ++) {
        /**< Generate the ith permutation */
        for(j = 0; j < 256; j ++) {
            res[i].P[j] = j;
        }

        /**< The 1st permutation returned is an identity permutation */
        if(i == 0) {
            continue;
        }

        for(j = 255; j >= 0; j --) {
            int k = rand() % (j + 1);
            uint8_t temp = res[i].P[j];
            res[i].P[j] = res[i].P[k];
            res[i].P[k] = temp;
        }
    }

    return res;
}
