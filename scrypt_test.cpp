#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

#define SHA_LONG unsigned long
#define MD32_REG_T int
#define SHA_LBLOCK  16

#define SHA256_DIGEST_LENGTH    32
#define DATA_ORDER_IS_BIG_ENDIAN

#define HASH_LONG       SHA_LONG
#define HASH_CTX        SHA256_CTX
#define HASH_CBLOCK     SHA_CBLOCK

#define SHA_CBLOCK  (SHA_LBLOCK*4)

#define SHA256_CBLOCK   (SHA_LBLOCK*4) 

#define ROTATE(a,n)     (((a)<<(n))|(((a)&0xffffffff)>>(32-(n))))



#ifdef DATA_ORDER_IS_BIG_ENDIAN


#ifndef HOST_c2l
#define HOST_c2l(c,l)   (l =(((unsigned long)(*((c)++)))<<24),      \
             l|=(((unsigned long)(*((c)++)))<<16),      \
             l|=(((unsigned long)(*((c)++)))<< 8),      \
             l|=(((unsigned long)(*((c)++)))    ),      \
             l)
#endif
#ifndef HOST_l2c
#define HOST_l2c(l,c)   (*((c)++)=(unsigned char)(((l)>>24)&0xff),  \
             *((c)++)=(unsigned char)(((l)>>16)&0xff),  \
             *((c)++)=(unsigned char)(((l)>> 8)&0xff),  \
             *((c)++)=(unsigned char)(((l)    )&0xff),  \
             l)
#endif



#elif defined(DATA_ORDER_IS_LITTLE_ENDIAN)

#ifndef HOST_c2l

#define HOST_c2l(c,l)   (l =(((unsigned long)(*((c)++)))    ),      \
             l|=(((unsigned long)(*((c)++)))<< 8),      \
             l|=(((unsigned long)(*((c)++)))<<16),      \
             l|=(((unsigned long)(*((c)++)))<<24),      \
             l)
#endif
#ifndef HOST_l2c
#define HOST_l2c(l,c)   (*((c)++)=(unsigned char)(((l)    )&0xff),  \
             *((c)++)=(unsigned char)(((l)>> 8)&0xff),  \
             *((c)++)=(unsigned char)(((l)>>16)&0xff),  \
             *((c)++)=(unsigned char)(((l)>>24)&0xff),  \
             l)
#endif

#endif


#define Sigma0(x)   (ROTATE((x),30) ^ ROTATE((x),19) ^ ROTATE((x),10))
#define Sigma1(x)   (ROTATE((x),26) ^ ROTATE((x),21) ^ ROTATE((x),7))
#define sigma0(x)   (ROTATE((x),25) ^ ROTATE((x),14) ^ ((x)>>3))
#define sigma1(x)   (ROTATE((x),15) ^ ROTATE((x),13) ^ ((x)>>10))

#define Ch(x,y,z)   (((x) & (y)) ^ ((~(x)) & (z)))
#define Maj(x,y,z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

static const SHA_LONG K256[64] = {
    0x428a2f98UL,0x71374491UL,0xb5c0fbcfUL,0xe9b5dba5UL,
    0x3956c25bUL,0x59f111f1UL,0x923f82a4UL,0xab1c5ed5UL,
    0xd807aa98UL,0x12835b01UL,0x243185beUL,0x550c7dc3UL,
    0x72be5d74UL,0x80deb1feUL,0x9bdc06a7UL,0xc19bf174UL,
    0xe49b69c1UL,0xefbe4786UL,0x0fc19dc6UL,0x240ca1ccUL,
    0x2de92c6fUL,0x4a7484aaUL,0x5cb0a9dcUL,0x76f988daUL,
    0x983e5152UL,0xa831c66dUL,0xb00327c8UL,0xbf597fc7UL,
    0xc6e00bf3UL,0xd5a79147UL,0x06ca6351UL,0x14292967UL,
    0x27b70a85UL,0x2e1b2138UL,0x4d2c6dfcUL,0x53380d13UL,
    0x650a7354UL,0x766a0abbUL,0x81c2c92eUL,0x92722c85UL,
    0xa2bfe8a1UL,0xa81a664bUL,0xc24b8b70UL,0xc76c51a3UL,
    0xd192e819UL,0xd6990624UL,0xf40e3585UL,0x106aa070UL,
    0x19a4c116UL,0x1e376c08UL,0x2748774cUL,0x34b0bcb5UL,
    0x391c0cb3UL,0x4ed8aa4aUL,0x5b9cca4fUL,0x682e6ff3UL,
    0x748f82eeUL,0x78a5636fUL,0x84c87814UL,0x8cc70208UL,
    0x90befffaUL,0xa4506cebUL,0xbef9a3f7UL,0xc67178f2UL };



typedef struct SHA256state_st
    {
    SHA_LONG h[8];
    SHA_LONG Nl,Nh;
    SHA_LONG data[SHA_LBLOCK];
    unsigned int num,md_len;
    } SHA256_CTX;



int SHA256_Init (SHA256_CTX *c)
    {
    memset (c,0,sizeof(*c));
    c->h[0]=0x6a09e667UL;   c->h[1]=0xbb67ae85UL;
    c->h[2]=0x3c6ef372UL;   c->h[3]=0xa54ff53aUL;
    c->h[4]=0x510e527fUL;   c->h[5]=0x9b05688cUL;
    c->h[6]=0x1f83d9abUL;   c->h[7]=0x5be0cd19UL;
    c->md_len=SHA256_DIGEST_LENGTH;
    return 1;
    }

#define HASH_UPDATE     SHA256_Update
#define HASH_TRANSFORM      SHA256_Transform
#define HASH_FINAL      SHA256_Final
#define HASH_BLOCK_DATA_ORDER   sha256_block_data_order

#define ROUND_00_15(i,a,b,c,d,e,f,g,h)      do {    \
    T1 += h + Sigma1(e) + Ch(e,f,g) + K256[i];  \
    h = Sigma0(a) + Maj(a,b,c);         \
    d += T1;    h += T1;        } while (0)

#define ROUND_16_63(i,a,b,c,d,e,f,g,h,X)    do {    \
    s0 = X[(i+1)&0x0f]; s0 = sigma0(s0);    \
    s1 = X[(i+14)&0x0f];    s1 = sigma1(s1);    \
    T1 = X[(i)&0x0f] += s0 + s1 + X[(i+9)&0x0f];    \
    ROUND_00_15(i,a,b,c,d,e,f,g,h);     } while (0)

static void sha256_block_data_order (SHA256_CTX *ctx, const void *in, size_t num)
    {
	printf("%s %d\n", __PRETTY_FUNCTION__, num);
    unsigned MD32_REG_T a,b,c,d,e,f,g,h,s0,s1,T1;
    SHA_LONG    X[16];
    int i;
    const unsigned char *data=(const unsigned char*)in;
    const union { long one; char little; } is_endian = {1};

            while (num--) {

    a = ctx->h[0];  b = ctx->h[1];  c = ctx->h[2];  d = ctx->h[3];
    e = ctx->h[4];  f = ctx->h[5];  g = ctx->h[6];  h = ctx->h[7];

    if (!is_endian.little && sizeof(SHA_LONG)==4 && ((size_t)in%4)==0)
        {
        const SHA_LONG *W=(const SHA_LONG *)data;

        T1 = X[0] = W[0];   ROUND_00_15(0,a,b,c,d,e,f,g,h);
        T1 = X[1] = W[1];   ROUND_00_15(1,h,a,b,c,d,e,f,g);
        T1 = X[2] = W[2];   ROUND_00_15(2,g,h,a,b,c,d,e,f);
        T1 = X[3] = W[3];   ROUND_00_15(3,f,g,h,a,b,c,d,e);
        T1 = X[4] = W[4];   ROUND_00_15(4,e,f,g,h,a,b,c,d);
        T1 = X[5] = W[5];   ROUND_00_15(5,d,e,f,g,h,a,b,c);
        T1 = X[6] = W[6];   ROUND_00_15(6,c,d,e,f,g,h,a,b);
        T1 = X[7] = W[7];   ROUND_00_15(7,b,c,d,e,f,g,h,a);
        T1 = X[8] = W[8];   ROUND_00_15(8,a,b,c,d,e,f,g,h);
        T1 = X[9] = W[9];   ROUND_00_15(9,h,a,b,c,d,e,f,g);
        T1 = X[10] = W[10]; ROUND_00_15(10,g,h,a,b,c,d,e,f);
        T1 = X[11] = W[11]; ROUND_00_15(11,f,g,h,a,b,c,d,e);
        T1 = X[12] = W[12]; ROUND_00_15(12,e,f,g,h,a,b,c,d);
        T1 = X[13] = W[13]; ROUND_00_15(13,d,e,f,g,h,a,b,c);
        T1 = X[14] = W[14]; ROUND_00_15(14,c,d,e,f,g,h,a,b);
        T1 = X[15] = W[15]; ROUND_00_15(15,b,c,d,e,f,g,h,a);

        data += SHA256_CBLOCK;
        }
    else
    {
        SHA_LONG l;

        HOST_c2l(data,l); T1 = X[0] = l;  ROUND_00_15(0,a,b,c,d,e,f,g,h);
        HOST_c2l(data,l); T1 = X[1] = l;  ROUND_00_15(1,h,a,b,c,d,e,f,g);
        HOST_c2l(data,l); T1 = X[2] = l;  ROUND_00_15(2,g,h,a,b,c,d,e,f);
        HOST_c2l(data,l); T1 = X[3] = l;  ROUND_00_15(3,f,g,h,a,b,c,d,e);
        HOST_c2l(data,l); T1 = X[4] = l;  ROUND_00_15(4,e,f,g,h,a,b,c,d);
        HOST_c2l(data,l); T1 = X[5] = l;  ROUND_00_15(5,d,e,f,g,h,a,b,c);
        HOST_c2l(data,l); T1 = X[6] = l;  ROUND_00_15(6,c,d,e,f,g,h,a,b);
        HOST_c2l(data,l); T1 = X[7] = l;  ROUND_00_15(7,b,c,d,e,f,g,h,a);
        HOST_c2l(data,l); T1 = X[8] = l;  ROUND_00_15(8,a,b,c,d,e,f,g,h);
        HOST_c2l(data,l); T1 = X[9] = l;  ROUND_00_15(9,h,a,b,c,d,e,f,g);
        HOST_c2l(data,l); T1 = X[10] = l; ROUND_00_15(10,g,h,a,b,c,d,e,f);
        HOST_c2l(data,l); T1 = X[11] = l; ROUND_00_15(11,f,g,h,a,b,c,d,e);
        HOST_c2l(data,l); T1 = X[12] = l; ROUND_00_15(12,e,f,g,h,a,b,c,d);
        HOST_c2l(data,l); T1 = X[13] = l; ROUND_00_15(13,d,e,f,g,h,a,b,c);
        HOST_c2l(data,l); T1 = X[14] = l; ROUND_00_15(14,c,d,e,f,g,h,a,b);
        HOST_c2l(data,l); T1 = X[15] = l; ROUND_00_15(15,b,c,d,e,f,g,h,a);
        }

    for (i=16;i<64;i+=8)
        {
        ROUND_16_63(i+0,a,b,c,d,e,f,g,h,X);
        ROUND_16_63(i+1,h,a,b,c,d,e,f,g,X);
        ROUND_16_63(i+2,g,h,a,b,c,d,e,f,X);
        ROUND_16_63(i+3,f,g,h,a,b,c,d,e,X);
        ROUND_16_63(i+4,e,f,g,h,a,b,c,d,X);
        ROUND_16_63(i+5,d,e,f,g,h,a,b,c,X);
        ROUND_16_63(i+6,c,d,e,f,g,h,a,b,X);
        ROUND_16_63(i+7,b,c,d,e,f,g,h,a,X);
        }

    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;

            }
    }


int HASH_UPDATE (HASH_CTX *c, const void *data_, size_t len)
    {
    const unsigned char *data=(const unsigned char*)data_;
    unsigned char *p;
    HASH_LONG l;
    size_t n;

    if (len==0) return 1;

    l=(c->Nl+(((HASH_LONG)len)<<3))&0xffffffffUL;
    /* 95-05-24 eay Fixed a bug with the overflow handling, thanks to
     * Wei Dai <weidai@eskimo.com> for pointing it out. */
    if (l < c->Nl) /* overflow */
        c->Nh++;
    c->Nh+=(HASH_LONG)(len>>29);    /* might cause compiler warning on 16-bit */
    c->Nl=l;

    n = c->num;
    if (n != 0)
        {
        p=(unsigned char *)c->data;

        if (len >= HASH_CBLOCK || len+n >= HASH_CBLOCK)
            {
            memcpy (p+n,data,HASH_CBLOCK-n);
            HASH_BLOCK_DATA_ORDER (c,p,1);
            n      = HASH_CBLOCK-n;
            data  += n;
            len   -= n;
            c->num = 0;
            memset (p,0,HASH_CBLOCK);   /* keep it zeroed */
            }
        else
            {
            memcpy (p+n,data,len);
            c->num += (unsigned int)len;
            return 1;
            }
        }

    n = len/HASH_CBLOCK;
    if (n > 0)
        {
        HASH_BLOCK_DATA_ORDER (c,data,n);
        n    *= HASH_CBLOCK;
        data += n;
        len  -= n;
        }

    if (len != 0)
        {
        p = (unsigned char *)c->data;
        c->num = (unsigned int)len;
        memcpy (p,data,len);
        }
    return 1;
    }




void HASH_TRANSFORM (HASH_CTX *c, const unsigned char *data)
    {
    HASH_BLOCK_DATA_ORDER (c,data,1);
    }

#define HASH_MAKE_STRING(c,s)   do {    \
    unsigned long ll;       \
    unsigned int  nn;       \
    switch ((c)->md_len)        \
    {   case SHA256_DIGEST_LENGTH:  \
        for (nn=0;nn<SHA256_DIGEST_LENGTH/4;nn++)   \
        {   ll=(c)->h[nn]; HOST_l2c(ll,(s));   }    \
        break;          \
        default:            \
        if ((c)->md_len > SHA256_DIGEST_LENGTH) \
            return 0;               \
        for (nn=0;nn<(c)->md_len/4;nn++)        \
        {   ll=(c)->h[nn]; HOST_l2c(ll,(s));   }    \
        break;          \
    }               \
    } while (0)



int HASH_FINAL (unsigned char *md, HASH_CTX *c)
    {
    unsigned char *p = (unsigned char *)c->data;
    size_t n = c->num;

    p[n] = 0x80; /* there is always room for one */
    n++;

    if (n > (HASH_CBLOCK-8))
        {
        memset (p+n,0,HASH_CBLOCK-n);
        n=0;
        HASH_BLOCK_DATA_ORDER (c,p,1);
        }
    memset (p+n,0,HASH_CBLOCK-8-n);

    p += HASH_CBLOCK-8;
#if   defined(DATA_ORDER_IS_BIG_ENDIAN)
    (void)HOST_l2c(c->Nh,p);
    (void)HOST_l2c(c->Nl,p);
#elif defined(DATA_ORDER_IS_LITTLE_ENDIAN)
    (void)HOST_l2c(c->Nl,p);
    (void)HOST_l2c(c->Nh,p);
#endif
    p -= HASH_CBLOCK;
    HASH_BLOCK_DATA_ORDER (c,p,1);
    c->num=0;
    memset (p,0,HASH_CBLOCK);

#ifndef HASH_MAKE_STRING
#error "HASH_MAKE_STRING must be defined!"
#else
    HASH_MAKE_STRING(c,md);
#endif

    return 1;
    }

// ***********************************************************
//                SCRYPT
// ***********************************************************

static const int SCRYPT_SCRATCHPAD_SIZE = 131072 + 63;

inline uint32_t le32dec(const void *pp)
{
        const uint8_t *p = (uint8_t const *)pp;
        return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
            ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}

inline void le32enc(void *pp, uint32_t x)
{
        uint8_t *p = (uint8_t *)pp;
        p[0] = x & 0xff;
        p[1] = (x >> 8) & 0xff;
        p[2] = (x >> 16) & 0xff;
        p[3] = (x >> 24) & 0xff;
}

inline uint32_t be32dec(const void *pp)
{
        const uint8_t *p = (uint8_t const *)pp;
        return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
         ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}

inline void be32enc(void *pp, uint32_t x)
{
        uint8_t *p = (uint8_t *)pp;
        p[3] = x & 0xff;
        p[2] = (x >> 8) & 0xff;
        p[1] = (x >> 16) & 0xff;
        p[0] = (x >> 24) & 0xff;
}




#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

inline void xor_salsa8(uint32_t B[16], const uint32_t Bx[16])
{
        uint32_t x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15;
        int i;

        x00 = (B[ 0] ^= Bx[ 0]);
        x01 = (B[ 1] ^= Bx[ 1]);
        x02 = (B[ 2] ^= Bx[ 2]);
        x03 = (B[ 3] ^= Bx[ 3]);
        x04 = (B[ 4] ^= Bx[ 4]);
        x05 = (B[ 5] ^= Bx[ 5]);
        x06 = (B[ 6] ^= Bx[ 6]);
        x07 = (B[ 7] ^= Bx[ 7]);
        x08 = (B[ 8] ^= Bx[ 8]);
        x09 = (B[ 9] ^= Bx[ 9]);
        x10 = (B[10] ^= Bx[10]);
        x11 = (B[11] ^= Bx[11]);
        x12 = (B[12] ^= Bx[12]);
        x13 = (B[13] ^= Bx[13]);
        x14 = (B[14] ^= Bx[14]);
        x15 = (B[15] ^= Bx[15]);
        xor_salsa8_label1:for (i = 0; i < 8; i += 2) {
                /* Operate on columns. */
                x04 ^= ROTL(x00 + x12, 7); x09 ^= ROTL(x05 + x01, 7);
                x14 ^= ROTL(x10 + x06, 7); x03 ^= ROTL(x15 + x11, 7);

                x08 ^= ROTL(x04 + x00, 9); x13 ^= ROTL(x09 + x05, 9);
                x02 ^= ROTL(x14 + x10, 9); x07 ^= ROTL(x03 + x15, 9);

                x12 ^= ROTL(x08 + x04, 13); x01 ^= ROTL(x13 + x09, 13);
                x06 ^= ROTL(x02 + x14, 13); x11 ^= ROTL(x07 + x03, 13);

                x00 ^= ROTL(x12 + x08, 18); x05 ^= ROTL(x01 + x13, 18);
                x10 ^= ROTL(x06 + x02, 18); x15 ^= ROTL(x11 + x07, 18);

                /* Operate on rows. */
                x01 ^= ROTL(x00 + x03, 7); x06 ^= ROTL(x05 + x04, 7);
                x11 ^= ROTL(x10 + x09, 7); x12 ^= ROTL(x15 + x14, 7);

                x02 ^= ROTL(x01 + x00, 9); x07 ^= ROTL(x06 + x05, 9);
                x08 ^= ROTL(x11 + x10, 9); x13 ^= ROTL(x12 + x15, 9);

                x03 ^= ROTL(x02 + x01, 13); x04 ^= ROTL(x07 + x06, 13);
                x09 ^= ROTL(x08 + x11, 13); x14 ^= ROTL(x13 + x12, 13);

                x00 ^= ROTL(x03 + x02, 18); x05 ^= ROTL(x04 + x07, 18);
                x10 ^= ROTL(x09 + x08, 18); x15 ^= ROTL(x14 + x13, 18);
        }
        B[ 0] += x00;
        B[ 1] += x01;
        B[ 2] += x02;
        B[ 3] += x03;
        B[ 4] += x04;
        B[ 5] += x05;
        B[ 6] += x06;
        B[ 7] += x07;
        B[ 8] += x08;
        B[ 9] += x09;
        B[10] += x10;
        B[11] += x11;
        B[12] += x12;
        B[13] += x13;
        B[14] += x14;
        B[15] += x15;
}

typedef struct HMAC_SHA256Context {
        SHA256_CTX ictx;
        SHA256_CTX octx;
} HMAC_SHA256_CTX;

/* Initialize an HMAC-SHA256 operation with the given key. */
static void
HMAC_SHA256_Init(HMAC_SHA256_CTX *ctx, const void *_K, size_t Klen)
{
        unsigned char pad[64];
        unsigned char khash[32];
        const unsigned char *K = (const unsigned char *)_K;
        size_t i;

        /* If Klen > 64, the key is really SHA256(K). */
        if (Klen > 64) {
                SHA256_Init(&ctx->ictx);
                SHA256_Update(&ctx->ictx, K, Klen);
                SHA256_Final(khash, &ctx->ictx);
                K = khash;
                Klen = 32;
        }

        /* Inner SHA256 operation is SHA256(K xor [block of 0x36] || data). */
        SHA256_Init(&ctx->ictx);
        memset(pad, 0x36, 64);
        for (i = 0; i < Klen; i++)
                pad[i] ^= K[i];
        SHA256_Update(&ctx->ictx, pad, 64);

        /* Outer SHA256 operation is SHA256(K xor [block of 0x5c] || hash). */
        SHA256_Init(&ctx->octx);
        memset(pad, 0x5c, 64);
        for (i = 0; i < Klen; i++)
                pad[i] ^= K[i];
        SHA256_Update(&ctx->octx, pad, 64);

        /* Clean the stack. */
        memset(khash, 0, 32);
}

/* Add bytes to the HMAC-SHA256 operation. */
static void
HMAC_SHA256_Update(HMAC_SHA256_CTX *ctx, const void *in, size_t len)
{
        /* Feed data to the inner SHA256 operation. */
        SHA256_Update(&ctx->ictx, in, len);
}

/* Finish an HMAC-SHA256 operation. */
static void
HMAC_SHA256_Final(unsigned char digest[32], HMAC_SHA256_CTX *ctx)
{
        unsigned char ihash[32];

        /* Finish the inner SHA256 operation. */
        SHA256_Final(ihash, &ctx->ictx);

        /* Feed the inner hash to the outer SHA256 operation. */
        SHA256_Update(&ctx->octx, ihash, 32);

        /* Finish the outer SHA256 operation. */
        SHA256_Final(digest, &ctx->octx);

        /* Clean the stack. */
        memset(ihash, 0, 32);
}

/**
* PBKDF2_SHA256(passwd, passwdlen, salt, saltlen, c, buf, dkLen):
* Compute PBKDF2(passwd, salt, c, dkLen) using HMAC-SHA256 as the PRF, and
* write the output to buf. The value dkLen must be at most 32 * (2^32 - 1).
*/
void
PBKDF2_SHA256(const uint8_t *passwd, size_t passwdlen, const uint8_t *salt,
    size_t saltlen, uint64_t c, uint8_t *buf, size_t dkLen)
{
        HMAC_SHA256_CTX PShctx, hctx;
        size_t i;
        uint8_t ivec[4];
        uint8_t U[32];
        uint8_t T[32];
        uint64_t j;
        int k;
        size_t clen;

        /* Compute HMAC state after processing P and S. */
        HMAC_SHA256_Init(&PShctx, passwd, passwdlen);
        HMAC_SHA256_Update(&PShctx, salt, saltlen);

        /* Iterate through the blocks. */
        for (i = 0; i * 32 < dkLen; i++) {
                /* Generate INT(i + 1). */
                be32enc(ivec, (uint32_t)(i + 1));

                /* Compute U_1 = PRF(P, S || INT(i)). */
                memcpy(&hctx, &PShctx, sizeof(HMAC_SHA256_CTX));
                HMAC_SHA256_Update(&hctx, ivec, 4);
                HMAC_SHA256_Final(U, &hctx);

                /* T_i = U_1 ... */
                memcpy(T, U, 32);

                for (j = 2; j <= c; j++) {
                        /* Compute U_j. */
                        HMAC_SHA256_Init(&hctx, passwd, passwdlen);
                        HMAC_SHA256_Update(&hctx, U, 32);
                        HMAC_SHA256_Final(U, &hctx);

                        /* ... xor U_j ... */
                        for (k = 0; k < 32; k++)
                                T[k] ^= U[k];
                }

                /* Copy as many bytes as necessary into buf. */
                clen = dkLen - i * 32;
                if (clen > 32)
                        clen = 32;
                memcpy(&buf[i * 32], T, clen);
        }

        /* Clean PShctx, since we never called _Final on it. */
        memset(&PShctx, 0, sizeof(HMAC_SHA256_CTX));
}

void scrypt_1024_1_1_256_sp_generic(const char *input, char *output, char *scratchpad)
{
        uint8_t B[128];
        uint32_t X[32];
        uint32_t *V;//[128*1024/4];
        uint32_t i, j, k;


        V = (uint32_t *)(((uintptr_t)(scratchpad) + 63) & ~ (uintptr_t)(63));
		//V=(uint32_t*)scratchpad;

        PBKDF2_SHA256((const uint8_t *)input, 80, (const uint8_t *)input, 80, 1, B, 128);


        for (k = 0; k < 32; k++)
                X[k] = le32dec(&B[4 * k]);


        for (i = 0; i < 1024; i++) {
                memcpy(&V[i * 32], (const uint32_t*)X, 128);
                xor_salsa8(&X[0], &X[16]);
                xor_salsa8(&X[16], &X[0]);
        }
        for (i = 0; i < 1024; i++) {
                j = 32 * (X[16] & 1023);
                for (k = 0; k < 32; k++)
                        X[k] ^= V[j + k];
                xor_salsa8(&X[0], &X[16]);
                xor_salsa8(&X[16], &X[0]);
        }

        for (k = 0; k < 32; k++)
                le32enc(&B[4 * k], X[k]);

        PBKDF2_SHA256((const uint8_t *)input, 80, B, 128, 1, (uint8_t *)output, 32);
}

void scrypt_1024_1_1_256(const char *input, char *output)
{
        char scratchpad[SCRYPT_SCRATCHPAD_SIZE];
    scrypt_1024_1_1_256_sp_generic(input, output, scratchpad);
}

int main(int argc, char** argv) {
	const char in[80]={(char)0x02,(char)0x00,(char)0x00,(char)0x00,(char)0xea,(char)0xa6,(char)0xa6,(char)0x3e,(char)0xf4,(char)0xd4,(char)0x24,(char)0xe0,(char)0x02,
						(char)0x3d,(char)0xb3,(char)0x49,(char)0xe9,(char)0x5b,(char)0xb2,(char)0xc1,(char)0x5e,(char)0xdc,(char)0x15,(char)0x23,(char)0xad,(char)0xf6,(char)0x6f,
						(char)0x1d,(char)0x3d,(char)0x6f,(char)0x48,(char)0xd0,(char)0xc1,(char)0xdd,(char)0x1c,(char)0x9c,(char)0xf7,(char)0x13,(char)0x4c,(char)0xf0,(char)0x7c,(char)0x2f,
						(char)0x6e,(char)0xde,(char)0x69,(char)0x1b,(char)0x8c,(char)0x95,(char)0x6a,(char)0x7a,(char)0x02,(char)0xfb,(char)0x84,(char)0xe3,(char)0xd7,(char)0xb6,(char)0xcc,(char)0x7d,
						(char)0x51,(char)0x37,(char)0xae,(char)0xa2,(char)0x34,(char)0x38,(char)0x46,(char)0x09,(char)0xab,(char)0x5a,(char)0xd0,(char)0x17,(char)0xe0,(char)0x52,(char)0x1b,(char)0x64,(char)0x10,
						(char)0x1b,(char)0x00,(char)0x33,(char)0x12,(char)0x31};
	unsigned char out[1024]={0,};
	scrypt_1024_1_1_256(in,(char*)out);
//	write(1,out,32);
	for(int i=0;i<32;i++)
		printf("%02x", out[i]);
	printf("\n");
	return 0;
}
