#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>

#include "packet_lengths.h"
//#include "headers.h"

#include "packet.h"
#define IPSEC_REG_NUM 20
//#include "../../common/para.h"


double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

void DES_ROUND(uint32 *SK, uint32 X, uint32 Y)
{
    uint32 T;

    T = *SK ^ X;
    Y ^= SB8[ (T      ) & 0x3F ] ^
         SB6[ (T >>  8) & 0x3F ] ^
         SB4[ (T >> 16) & 0x3F ] ^
         SB2[ (T >> 24) & 0x3F ];

    T = *SK++ ^ ((X << 28) | (X >> 4));
    Y ^= SB7[ (T      ) & 0x3F ] ^
         SB5[ (T >>  8) & 0x3F ] ^
         SB3[ (T >> 16) & 0x3F ] ^
         SB1[ (T >> 24) & 0x3F ];
}
__device__ void DES_ROUND_dev(uint32 *SK, uint32 X, uint32 Y)
{
    uint32 T;

    T = *SK ^ X;
    Y ^= SB8[ (T      ) & 0x3F ] ^
         SB6[ (T >>  8) & 0x3F ] ^
         SB4[ (T >> 16) & 0x3F ] ^
         SB2[ (T >> 24) & 0x3F ];

    T = *SK++ ^ ((X << 28) | (X >> 4));
    Y ^= SB7[ (T      ) & 0x3F ] ^
         SB5[ (T >>  8) & 0x3F ] ^
         SB3[ (T >> 16) & 0x3F ] ^
         SB1[ (T >> 24) & 0x3F ];
}
/* DES key schedule */

int des_main_ks( uint32 *SK, uint8 *key )
{
    int i;
    uint32 X, Y, T;

    GET_UINT32( X, key, 0 );
    GET_UINT32( Y, key, 4 );

    /* Permuted Choice 1 */

    T =  ((Y >>  4) ^ X) & 0x0F0F0F0F;  X ^= T; Y ^= (T <<  4);
    T =  ((Y      ) ^ X) & 0x10101010;  X ^= T; Y ^= (T      );

    X =   (LHs[ (X      ) & 0xF] << 3) | (LHs[ (X >>  8) & 0xF ] << 2)
        | (LHs[ (X >> 16) & 0xF] << 1) | (LHs[ (X >> 24) & 0xF ]     )
        | (LHs[ (X >>  5) & 0xF] << 7) | (LHs[ (X >> 13) & 0xF ] << 6)
        | (LHs[ (X >> 21) & 0xF] << 5) | (LHs[ (X >> 29) & 0xF ] << 4);

    Y =   (RHs[ (Y >>  1) & 0xF] << 3) | (RHs[ (Y >>  9) & 0xF ] << 2)
        | (RHs[ (Y >> 17) & 0xF] << 1) | (RHs[ (Y >> 25) & 0xF ]     )
        | (RHs[ (Y >>  4) & 0xF] << 7) | (RHs[ (Y >> 12) & 0xF ] << 6)
        | (RHs[ (Y >> 20) & 0xF] << 5) | (RHs[ (Y >> 28) & 0xF ] << 4);

    X &= 0x0FFFFFFF;
    Y &= 0x0FFFFFFF;

    /* calculate subkeys */

    for( i = 0; i < 16; i++ )
    {
        if( i < 2 || i == 8 || i == 15 )
        {
            X = ((X <<  1) | (X >> 27)) & 0x0FFFFFFF;
            Y = ((Y <<  1) | (Y >> 27)) & 0x0FFFFFFF;
        }
        else
        {
            X = ((X <<  2) | (X >> 26)) & 0x0FFFFFFF;
            Y = ((Y <<  2) | (Y >> 26)) & 0x0FFFFFFF;
        }

        *SK++ =   ((X <<  4) & 0x24000000) | ((X << 28) & 0x10000000)
                | ((X << 14) & 0x08000000) | ((X << 18) & 0x02080000)
                | ((X <<  6) & 0x01000000) | ((X <<  9) & 0x00200000)
                | ((X >>  1) & 0x00100000) | ((X << 10) & 0x00040000)
                | ((X <<  2) & 0x00020000) | ((X >> 10) & 0x00010000)
                | ((Y >> 13) & 0x00002000) | ((Y >>  4) & 0x00001000)
                | ((Y <<  6) & 0x00000800) | ((Y >>  1) & 0x00000400)
                | ((Y >> 14) & 0x00000200) | ((Y      ) & 0x00000100)
                | ((Y >>  5) & 0x00000020) | ((Y >> 10) & 0x00000010)
                | ((Y >>  3) & 0x00000008) | ((Y >> 18) & 0x00000004)
                | ((Y >> 26) & 0x00000002) | ((Y >> 24) & 0x00000001);

        *SK++ =   ((X << 15) & 0x20000000) | ((X << 17) & 0x10000000)
                | ((X << 10) & 0x08000000) | ((X << 22) & 0x04000000)
                | ((X >>  2) & 0x02000000) | ((X <<  1) & 0x01000000)
                | ((X << 16) & 0x00200000) | ((X << 11) & 0x00100000)
                | ((X <<  3) & 0x00080000) | ((X >>  6) & 0x00040000)
                | ((X << 15) & 0x00020000) | ((X >>  4) & 0x00010000)
                | ((Y >>  2) & 0x00002000) | ((Y <<  8) & 0x00001000)
                | ((Y >> 14) & 0x00000808) | ((Y >>  9) & 0x00000400)
                | ((Y      ) & 0x00000200) | ((Y <<  7) & 0x00000100)
                | ((Y >>  7) & 0x00000020) | ((Y >>  3) & 0x00000011)
                | ((Y <<  2) & 0x00000004) | ((Y >> 21) & 0x00000002);
    }

    return( 0 );
}

int des_set_key( uint32 *esk, uint32 *dsk, uint8 key1[8],
                                uint8 key2[8], uint8 key3[8])
{
    int i;

    /* setup encryption subkeys */

    des_main_ks( esk, key1 );
    des_main_ks( dsk + 32, key2 );
    des_main_ks( esk + 64, key3 );



    /* setup decryption subkeys */

    for( i = 0; i < 32; i += 2 )
    {
        dsk[i] = esk[94 - i];
        dsk[i + 1] = esk[95 - i];
        esk[i + 32] = dsk[62 - i];
        esk[i + 33] = dsk[63 - i];
        dsk[i + 64] = esk[30 - i];
        dsk[i + 65] = esk[31 - i];

    }

    return( 0 );
}

/* DES 64-bit block encryption/decryption */

void des_crypt( uint32 *SK, uint8 *input, uint8 *output, int len)
{
    int i;
    uint32 X, Y, T;

    for(i = 0; i < len; i++){
      X = ( (uint32) *(input + i*8) << 24 )
        | ( (uint32) *(input + (i*8) + 1) << 16 )
        | ( (uint32) *(input + (i*8) + 2) <<  8 )
        | ( (uint32) *(input + (i*8) + 3)       );

      Y = ( (uint32) *(input + (i*8) + 4) << 24 )
        | ( (uint32) *(input + (i*8) + 5) << 16 )
        | ( (uint32) *(input + (i*8) + 6) <<  8 )
        | ( (uint32) *(input + (i*8) + 7)       );

      DES_IP( X, Y );

      DES_ROUND( SK, Y, X );  DES_ROUND( (SK + 2), X, Y );
      DES_ROUND( (SK + 4), Y, X );  DES_ROUND( (SK + 6), X, Y );
      DES_ROUND( (SK + 8), Y, X );  DES_ROUND( (SK + 10), X, Y );
      DES_ROUND( (SK + 12), Y, X );  DES_ROUND( (SK + 14), X, Y );
      DES_ROUND( (SK + 16), Y, X );  DES_ROUND( (SK + 18), X, Y );
      DES_ROUND( (SK + 20), Y, X );  DES_ROUND( (SK + 22), X, Y );
      DES_ROUND( (SK + 24), Y, X );  DES_ROUND( (SK + 26), X, Y );
      DES_ROUND( (SK + 28), Y, X );  DES_ROUND( (SK + 30), X, Y );

      DES_ROUND( (SK + 32), X, Y );  DES_ROUND( (SK + 34), Y, X );
      DES_ROUND( (SK + 36), X, Y );  DES_ROUND( (SK + 38), Y, X );
      DES_ROUND( (SK + 40), X, Y );  DES_ROUND( (SK + 42), Y, X );
      DES_ROUND( (SK + 44), X, Y );  DES_ROUND( (SK + 46), Y, X );
      DES_ROUND( (SK + 48), X, Y );  DES_ROUND( (SK + 50), Y, X );
      DES_ROUND( (SK + 52), X, Y );  DES_ROUND( (SK + 54), Y, X );
      DES_ROUND( (SK + 56), X, Y );  DES_ROUND( (SK + 58), Y, X );
      DES_ROUND( (SK + 60), X, Y );  DES_ROUND( (SK + 62), Y, X );

      DES_ROUND( (SK + 64), Y, X );  DES_ROUND( (SK + 66), X, Y );
      DES_ROUND( (SK + 68), Y, X );  DES_ROUND( (SK + 70), X, Y );
      DES_ROUND( (SK + 72), Y, X );  DES_ROUND( (SK + 74), X, Y );
      DES_ROUND( (SK + 76), Y, X );  DES_ROUND( (SK + 78), X, Y );
      DES_ROUND( (SK + 80), Y, X );  DES_ROUND( (SK + 82), X, Y );
      DES_ROUND( (SK + 84), Y, X );  DES_ROUND( (SK + 86), X, Y );
      DES_ROUND( (SK + 88), Y, X );  DES_ROUND( (SK + 90), X, Y );
      DES_ROUND( (SK + 92), Y, X );  DES_ROUND( (SK + 94), X, Y );

      DES_FP( Y, X );

      *(output + i*8) = (uint8) ( (Y) >> 24 );
      *(output + (i*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + (i*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + (i*8) + 3) = (uint8) ( (Y)       );
      *(output + (i*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + (i*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + (i*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + (i*8) + 7) = (uint8) ( (X)       );
	
    }
}

__device__ void des_crypt_dev( uint32 *SK, uint8 *input, uint8 *output, int len, int thread)
{
    int i;
    uint32 X, Y, T;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < thread){
      for(i = 0; i < len/thread; i++){
      //  i
	      X = ( (uint32) *(input + (i*thread+tid)*8) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 1) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 2) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 3)       );

        Y = ( (uint32) *(input + ((i*thread+tid)*8) + 4) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 5) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 6) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 7)       );


        DES_IP( X, Y );

        DES_ROUND_dev( SK, Y, X );  DES_ROUND_dev( (SK + 2), X, Y );
        DES_ROUND_dev( (SK + 4), Y, X );  DES_ROUND_dev( (SK + 6), X, Y );
        DES_ROUND_dev( (SK + 8), Y, X );  DES_ROUND_dev( (SK + 10), X, Y );
        DES_ROUND_dev( (SK + 12), Y, X );  DES_ROUND_dev( (SK + 14), X, Y );
        DES_ROUND_dev( (SK + 16), Y, X );  DES_ROUND_dev( (SK + 18), X, Y );
        DES_ROUND_dev( (SK + 20), Y, X );  DES_ROUND_dev( (SK + 22), X, Y );
        DES_ROUND_dev( (SK + 24), Y, X );  DES_ROUND_dev( (SK + 26), X, Y );
        DES_ROUND_dev( (SK + 28), Y, X );  DES_ROUND_dev( (SK + 30), X, Y );

        DES_ROUND_dev( (SK + 32), X, Y );  DES_ROUND_dev( (SK + 34), Y, X );
        DES_ROUND_dev( (SK + 36), X, Y );  DES_ROUND_dev( (SK + 38), Y, X );
        DES_ROUND_dev( (SK + 40), X, Y );  DES_ROUND_dev( (SK + 42), Y, X );
        DES_ROUND_dev( (SK + 44), X, Y );  DES_ROUND_dev( (SK + 46), Y, X );
        DES_ROUND_dev( (SK + 48), X, Y );  DES_ROUND_dev( (SK + 50), Y, X );
        DES_ROUND_dev( (SK + 52), X, Y );  DES_ROUND_dev( (SK + 54), Y, X );
        DES_ROUND_dev( (SK + 56), X, Y );  DES_ROUND_dev( (SK + 58), Y, X );
        DES_ROUND_dev( (SK + 60), X, Y );  DES_ROUND_dev( (SK + 62), Y, X );

        DES_ROUND_dev( (SK + 64), Y, X );  DES_ROUND_dev( (SK + 66), X, Y );
        DES_ROUND_dev( (SK + 68), Y, X );  DES_ROUND_dev( (SK + 70), X, Y );
        DES_ROUND_dev( (SK + 72), Y, X );  DES_ROUND_dev( (SK + 74), X, Y );
        DES_ROUND_dev( (SK + 76), Y, X );  DES_ROUND_dev( (SK + 78), X, Y );
        DES_ROUND_dev( (SK + 80), Y, X );  DES_ROUND_dev( (SK + 82), X, Y );
        DES_ROUND_dev( (SK + 84), Y, X );  DES_ROUND_dev( (SK + 86), X, Y );
        DES_ROUND_dev( (SK + 88), Y, X );  DES_ROUND_dev( (SK + 90), X, Y );
        DES_ROUND_dev( (SK + 92), Y, X );  DES_ROUND_dev( (SK + 94), X, Y );

        DES_FP( Y, X );

      *(output + (i*thread+tid)*8) = (uint8) ( (Y) >> 24 );
      *(output + ((i*thread+tid)*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + ((i*thread+tid)*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + ((i*thread+tid)*8) + 3) = (uint8) ( (Y)       );
      *(output + ((i*thread+tid)*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + ((i*thread+tid)*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + ((i*thread+tid)*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + ((i*thread+tid)*8) + 7) = (uint8) ( (X)       );

      }
    }
}
// len= 256    16, thread =32
__global__ void des_encrypt_dev( uint32 *esk, uint32 *dsk, uint8 *input_o, uint8 *output_o, int len, int thread, int* reg_buffer, bool save_regs)
{
   // des_crypt_dev( esk, input, input, len, thread);
//uint32 *SK, uint8 *input, uint8 *output, int len, int thread
    uint32 *SK = esk;
    uint8 *input = input_o;
    uint8 *output= input_o;
    //len=32;
    //thread=32;
    int i;
    uint32 X, Y, T;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < thread){
      for(i = 0; i < len/thread; i++){
      //  i
        X = ( (uint32) *(input + (i*thread+tid)*8) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 1) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 2) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 3)       );

        Y = ( (uint32) *(input + ((i*thread+tid)*8) + 4) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 5) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 6) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 7)       );


        DES_IP( X, Y );
        uint32 T1,X1,Y1;
        X1=Y;
        Y1=X;
        T1 = *SK ^ X1;
        Y1 ^= SB8[ (T1      ) & 0x3F ] ^
          SB6[ (T1 >>  8) & 0x3F ] ^
          SB4[ (T1 >> 16) & 0x3F ] ^
          SB2[ (T1 >> 24) & 0x3F ];

        T1 = *SK++ ^ ((X1 << 28) | (X1 >> 4));
        Y1 ^= SB7[ (T1      ) & 0x3F ] ^
          SB5[ (T1 >>  8) & 0x3F ] ^
          SB3[ (T1 >> 16) & 0x3F ] ^
          SB1[ (T1 >> 24) & 0x3F ];
        //DES_ROUND_dev( SK, Y, X ); 
        DES_ROUND_dev( (SK + 2), X, Y );
        DES_ROUND_dev( (SK + 4), Y, X );  DES_ROUND_dev( (SK + 6), X, Y );
        DES_ROUND_dev( (SK + 8), Y, X );  DES_ROUND_dev( (SK + 10), X, Y );
        DES_ROUND_dev( (SK + 12), Y, X );  DES_ROUND_dev( (SK + 14), X, Y );
        DES_ROUND_dev( (SK + 16), Y, X );  DES_ROUND_dev( (SK + 18), X, Y );
        DES_ROUND_dev( (SK + 20), Y, X );  DES_ROUND_dev( (SK + 22), X, Y );
        DES_ROUND_dev( (SK + 24), Y, X );  DES_ROUND_dev( (SK + 26), X, Y );
        DES_ROUND_dev( (SK + 28), Y, X );  DES_ROUND_dev( (SK + 30), X, Y );

        DES_ROUND_dev( (SK + 32), X, Y );  DES_ROUND_dev( (SK + 34), Y, X );
        DES_ROUND_dev( (SK + 36), X, Y );  DES_ROUND_dev( (SK + 38), Y, X );
        DES_ROUND_dev( (SK + 40), X, Y );  DES_ROUND_dev( (SK + 42), Y, X );
        DES_ROUND_dev( (SK + 44), X, Y );  DES_ROUND_dev( (SK + 46), Y, X );
        DES_ROUND_dev( (SK + 48), X, Y );  DES_ROUND_dev( (SK + 50), Y, X );
        DES_ROUND_dev( (SK + 52), X, Y );  DES_ROUND_dev( (SK + 54), Y, X );
        DES_ROUND_dev( (SK + 56), X, Y );  DES_ROUND_dev( (SK + 58), Y, X );
        DES_ROUND_dev( (SK + 60), X, Y );  DES_ROUND_dev( (SK + 62), Y, X );

        DES_ROUND_dev( (SK + 64), Y, X );  DES_ROUND_dev( (SK + 66), X, Y );
        DES_ROUND_dev( (SK + 68), Y, X );  DES_ROUND_dev( (SK + 70), X, Y );
        DES_ROUND_dev( (SK + 72), Y, X );  DES_ROUND_dev( (SK + 74), X, Y );
        DES_ROUND_dev( (SK + 76), Y, X );  DES_ROUND_dev( (SK + 78), X, Y );
        DES_ROUND_dev( (SK + 80), Y, X );  DES_ROUND_dev( (SK + 82), X, Y );
        DES_ROUND_dev( (SK + 84), Y, X );  DES_ROUND_dev( (SK + 86), X, Y );
        DES_ROUND_dev( (SK + 88), Y, X );  DES_ROUND_dev( (SK + 90), X, Y );
        DES_ROUND_dev( (SK + 92), Y, X );  DES_ROUND_dev( (SK + 94), X, Y );

        DES_FP( Y, X );

      *(output + (i*thread+tid)*8) = (uint8) ( (Y) >> 24 );
      *(output + ((i*thread+tid)*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + ((i*thread+tid)*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + ((i*thread+tid)*8) + 3) = (uint8) ( (Y)       );
      *(output + ((i*thread+tid)*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + ((i*thread+tid)*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + ((i*thread+tid)*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + ((i*thread+tid)*8) + 7) = (uint8) ( (X)       );

      }
    }

//    des_crypt_dev( dsk, input, output, len, thread);
//uint32 *SK, uint8 *input, uint8 *output, int len, int thread
    SK = dsk;
    input = input_o;
    output = output_o;

    
    if(tid < thread){
      for(i = 0; i < len/thread; i++){
      //  i
        X = ( (uint32) *(input + (i*thread+tid)*8) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 1) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 2) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 3)       );

        Y = ( (uint32) *(input + ((i*thread+tid)*8) + 4) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 5) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 6) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 7)       );


        DES_IP( X, Y );

        DES_ROUND_dev( SK, Y, X );  DES_ROUND_dev( (SK + 2), X, Y );
        DES_ROUND_dev( (SK + 4), Y, X );  DES_ROUND_dev( (SK + 6), X, Y );
        DES_ROUND_dev( (SK + 8), Y, X );  DES_ROUND_dev( (SK + 10), X, Y );
        DES_ROUND_dev( (SK + 12), Y, X );  DES_ROUND_dev( (SK + 14), X, Y );
        DES_ROUND_dev( (SK + 16), Y, X );  DES_ROUND_dev( (SK + 18), X, Y );
        DES_ROUND_dev( (SK + 20), Y, X );  DES_ROUND_dev( (SK + 22), X, Y );
        DES_ROUND_dev( (SK + 24), Y, X );  DES_ROUND_dev( (SK + 26), X, Y );
        DES_ROUND_dev( (SK + 28), Y, X );  DES_ROUND_dev( (SK + 30), X, Y );

        DES_ROUND_dev( (SK + 32), X, Y );  DES_ROUND_dev( (SK + 34), Y, X );
        DES_ROUND_dev( (SK + 36), X, Y );  DES_ROUND_dev( (SK + 38), Y, X );
        DES_ROUND_dev( (SK + 40), X, Y );  DES_ROUND_dev( (SK + 42), Y, X );
        DES_ROUND_dev( (SK + 44), X, Y );  DES_ROUND_dev( (SK + 46), Y, X );
        DES_ROUND_dev( (SK + 48), X, Y );  DES_ROUND_dev( (SK + 50), Y, X );
        DES_ROUND_dev( (SK + 52), X, Y );  DES_ROUND_dev( (SK + 54), Y, X );
        DES_ROUND_dev( (SK + 56), X, Y );  DES_ROUND_dev( (SK + 58), Y, X );
        DES_ROUND_dev( (SK + 60), X, Y );  DES_ROUND_dev( (SK + 62), Y, X );

        DES_ROUND_dev( (SK + 64), Y, X );  DES_ROUND_dev( (SK + 66), X, Y );
        DES_ROUND_dev( (SK + 68), Y, X );  DES_ROUND_dev( (SK + 70), X, Y );
        DES_ROUND_dev( (SK + 72), Y, X );  DES_ROUND_dev( (SK + 74), X, Y );
        DES_ROUND_dev( (SK + 76), Y, X );  DES_ROUND_dev( (SK + 78), X, Y );
        DES_ROUND_dev( (SK + 80), Y, X );  DES_ROUND_dev( (SK + 82), X, Y );
        DES_ROUND_dev( (SK + 84), Y, X );  DES_ROUND_dev( (SK + 86), X, Y );
        DES_ROUND_dev( (SK + 88), Y, X );  DES_ROUND_dev( (SK + 90), X, Y );
        DES_ROUND_dev( (SK + 92), Y, X );  DES_ROUND_dev( (SK + 94), X, Y );

        DES_FP( Y, X );

      *(output + (i*thread+tid)*8) = (uint8) ( (Y) >> 24 );
      *(output + ((i*thread+tid)*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + ((i*thread+tid)*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + ((i*thread+tid)*8) + 3) = (uint8) ( (Y)       );
      *(output + ((i*thread+tid)*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + ((i*thread+tid)*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + ((i*thread+tid)*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + ((i*thread+tid)*8) + 7) = (uint8) ( (X)       );

      }
    }


}

__global__ void des_encrypt_dev_save_regs( uint32 *esk, uint32 *dsk, uint8 *input_o, uint8 *output_o, int len, int thread, int* reg_buffer, bool save_regs)
{
   // des_crypt_dev( esk, input, input, len, thread);
//uint32 *SK, uint8 *input, uint8 *output, int len, int thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (save_regs) {
        //save regs
        for (int i=0; i<IPSEC_REG_NUM; i++) {
            //save_register(reg_buffer);
            reg_buffer[tid * IPSEC_REG_NUM + i] = tid;
        }
    }

    uint32 *SK = esk;
    uint8 *input = input_o;
    uint8 *output= input_o;
    //len=32;
    //thread=32;
    int i;
    uint32 X, Y, T;
    
    if(tid < thread){
      for(i = 0; i < len/thread; i++){
      //  i
        X = ( (uint32) *(input + (i*thread+tid)*8) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 1) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 2) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 3)       );

        Y = ( (uint32) *(input + ((i*thread+tid)*8) + 4) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 5) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 6) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 7)       );


        DES_IP( X, Y );
        uint32 T1,X1,Y1;
        X1=Y;
        Y1=X;
        T1 = *SK ^ X1;
        Y1 ^= SB8[ (T1      ) & 0x3F ] ^
          SB6[ (T1 >>  8) & 0x3F ] ^
          SB4[ (T1 >> 16) & 0x3F ] ^
          SB2[ (T1 >> 24) & 0x3F ];

        T1 = *SK++ ^ ((X1 << 28) | (X1 >> 4));
        Y1 ^= SB7[ (T1      ) & 0x3F ] ^
          SB5[ (T1 >>  8) & 0x3F ] ^
          SB3[ (T1 >> 16) & 0x3F ] ^
          SB1[ (T1 >> 24) & 0x3F ];
        //DES_ROUND_dev( SK, Y, X ); 
        DES_ROUND_dev( (SK + 2), X, Y );
        DES_ROUND_dev( (SK + 4), Y, X );  DES_ROUND_dev( (SK + 6), X, Y );
        DES_ROUND_dev( (SK + 8), Y, X );  DES_ROUND_dev( (SK + 10), X, Y );
        DES_ROUND_dev( (SK + 12), Y, X );  DES_ROUND_dev( (SK + 14), X, Y );
        DES_ROUND_dev( (SK + 16), Y, X );  DES_ROUND_dev( (SK + 18), X, Y );
        DES_ROUND_dev( (SK + 20), Y, X );  DES_ROUND_dev( (SK + 22), X, Y );
        DES_ROUND_dev( (SK + 24), Y, X );  DES_ROUND_dev( (SK + 26), X, Y );
        DES_ROUND_dev( (SK + 28), Y, X );  DES_ROUND_dev( (SK + 30), X, Y );

        DES_ROUND_dev( (SK + 32), X, Y );  DES_ROUND_dev( (SK + 34), Y, X );
        DES_ROUND_dev( (SK + 36), X, Y );  DES_ROUND_dev( (SK + 38), Y, X );
        DES_ROUND_dev( (SK + 40), X, Y );  DES_ROUND_dev( (SK + 42), Y, X );
        DES_ROUND_dev( (SK + 44), X, Y );  DES_ROUND_dev( (SK + 46), Y, X );
        DES_ROUND_dev( (SK + 48), X, Y );  DES_ROUND_dev( (SK + 50), Y, X );
        DES_ROUND_dev( (SK + 52), X, Y );  DES_ROUND_dev( (SK + 54), Y, X );
        DES_ROUND_dev( (SK + 56), X, Y );  DES_ROUND_dev( (SK + 58), Y, X );
        DES_ROUND_dev( (SK + 60), X, Y );  DES_ROUND_dev( (SK + 62), Y, X );

        DES_ROUND_dev( (SK + 64), Y, X );  DES_ROUND_dev( (SK + 66), X, Y );
        DES_ROUND_dev( (SK + 68), Y, X );  DES_ROUND_dev( (SK + 70), X, Y );
        DES_ROUND_dev( (SK + 72), Y, X );  DES_ROUND_dev( (SK + 74), X, Y );
        DES_ROUND_dev( (SK + 76), Y, X );  DES_ROUND_dev( (SK + 78), X, Y );
        DES_ROUND_dev( (SK + 80), Y, X );  DES_ROUND_dev( (SK + 82), X, Y );
        DES_ROUND_dev( (SK + 84), Y, X );  DES_ROUND_dev( (SK + 86), X, Y );
        DES_ROUND_dev( (SK + 88), Y, X );  DES_ROUND_dev( (SK + 90), X, Y );
        DES_ROUND_dev( (SK + 92), Y, X );  DES_ROUND_dev( (SK + 94), X, Y );

        DES_FP( Y, X );

      *(output + (i*thread+tid)*8) = (uint8) ( (Y) >> 24 );
      *(output + ((i*thread+tid)*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + ((i*thread+tid)*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + ((i*thread+tid)*8) + 3) = (uint8) ( (Y)       );
      *(output + ((i*thread+tid)*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + ((i*thread+tid)*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + ((i*thread+tid)*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + ((i*thread+tid)*8) + 7) = (uint8) ( (X)       );

      }
    }

//    des_crypt_dev( dsk, input, output, len, thread);
//uint32 *SK, uint8 *input, uint8 *output, int len, int thread
    SK = dsk;
    input = input_o;
    output = output_o;

    
    if(tid < thread){
      for(i = 0; i < len/thread; i++){
      //  i
        X = ( (uint32) *(input + (i*thread+tid)*8) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 1) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 2) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 3)       );

        Y = ( (uint32) *(input + ((i*thread+tid)*8) + 4) << 24 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 5) << 16 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 6) <<  8 )
        | ( (uint32) *(input + ((i*thread+tid)*8) + 7)       );


        DES_IP( X, Y );

        DES_ROUND_dev( SK, Y, X );  DES_ROUND_dev( (SK + 2), X, Y );
        DES_ROUND_dev( (SK + 4), Y, X );  DES_ROUND_dev( (SK + 6), X, Y );
        DES_ROUND_dev( (SK + 8), Y, X );  DES_ROUND_dev( (SK + 10), X, Y );
        DES_ROUND_dev( (SK + 12), Y, X );  DES_ROUND_dev( (SK + 14), X, Y );
        DES_ROUND_dev( (SK + 16), Y, X );  DES_ROUND_dev( (SK + 18), X, Y );
        DES_ROUND_dev( (SK + 20), Y, X );  DES_ROUND_dev( (SK + 22), X, Y );
        DES_ROUND_dev( (SK + 24), Y, X );  DES_ROUND_dev( (SK + 26), X, Y );
        DES_ROUND_dev( (SK + 28), Y, X );  DES_ROUND_dev( (SK + 30), X, Y );

        DES_ROUND_dev( (SK + 32), X, Y );  DES_ROUND_dev( (SK + 34), Y, X );
        DES_ROUND_dev( (SK + 36), X, Y );  DES_ROUND_dev( (SK + 38), Y, X );
        DES_ROUND_dev( (SK + 40), X, Y );  DES_ROUND_dev( (SK + 42), Y, X );
        DES_ROUND_dev( (SK + 44), X, Y );  DES_ROUND_dev( (SK + 46), Y, X );
        DES_ROUND_dev( (SK + 48), X, Y );  DES_ROUND_dev( (SK + 50), Y, X );
        DES_ROUND_dev( (SK + 52), X, Y );  DES_ROUND_dev( (SK + 54), Y, X );
        DES_ROUND_dev( (SK + 56), X, Y );  DES_ROUND_dev( (SK + 58), Y, X );
        DES_ROUND_dev( (SK + 60), X, Y );  DES_ROUND_dev( (SK + 62), Y, X );

        DES_ROUND_dev( (SK + 64), Y, X );  DES_ROUND_dev( (SK + 66), X, Y );
        DES_ROUND_dev( (SK + 68), Y, X );  DES_ROUND_dev( (SK + 70), X, Y );
        DES_ROUND_dev( (SK + 72), Y, X );  DES_ROUND_dev( (SK + 74), X, Y );
        DES_ROUND_dev( (SK + 76), Y, X );  DES_ROUND_dev( (SK + 78), X, Y );
        DES_ROUND_dev( (SK + 80), Y, X );  DES_ROUND_dev( (SK + 82), X, Y );
        DES_ROUND_dev( (SK + 84), Y, X );  DES_ROUND_dev( (SK + 86), X, Y );
        DES_ROUND_dev( (SK + 88), Y, X );  DES_ROUND_dev( (SK + 90), X, Y );
        DES_ROUND_dev( (SK + 92), Y, X );  DES_ROUND_dev( (SK + 94), X, Y );

        DES_FP( Y, X );

      *(output + (i*thread+tid)*8) = (uint8) ( (Y) >> 24 );
      *(output + ((i*thread+tid)*8) + 1) = (uint8) ( (Y) >> 16 );
      *(output + ((i*thread+tid)*8) + 2) = (uint8) ( (Y) >>  8 );
      *(output + ((i*thread+tid)*8) + 3) = (uint8) ( (Y)       );
      *(output + ((i*thread+tid)*8) + 4) = (uint8) ( (X) >> 24 );
      *(output + ((i*thread+tid)*8) + 5) = (uint8) ( (X) >> 16 );
      *(output + ((i*thread+tid)*8) + 6) = (uint8) ( (X) >>  8 );
      *(output + ((i*thread+tid)*8) + 7) = (uint8) ( (X)       );

      }
    }
    if (save_regs) {
        //save regs
        for (int i=0; i<IPSEC_REG_NUM; i++) {
            tid = reg_buffer[tid * IPSEC_REG_NUM + i];
        }
    }

}



void des_encrypt( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int len)
{
    des_crypt( esk, input, input, len);
    des_crypt( dsk, input, output, len);
}


/* For debugging purposes, to get the size of the packet_numberth packet */
unsigned int
packet_size (unsigned int packet_number)
{
  packet_number = packet_number % MAX_INDEX;
  return (packet_lengths[packet_number]);
}


