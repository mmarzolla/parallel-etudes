/****************************************************************************
 *
 * mpi-brute-force.c - Brute-force password cracking
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Brute-force password cracking
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-01-04

![DES cracker board developed in 1998 by the Electronic Frontier Foundation (EFF); this device can be used to brute-force a DES key. The original uploader was Matt Crypto at English Wikipedia. Later versions were uploaded by Ed g2s at en.wikipedia - CC BY 3.0 us, <https://commons.wikimedia.org/w/index.php?curid=2437815>](des-cracker.jpg)

The program [mpi-brute-force.c](mpi-brute-force.c) contains a 64-Byte
encrypted message stored in the array `enc[]`. The message has been
encrypted using the _XOR_ cryptographic algorithm, which applies the
"exclusive or" (xor) operator between a plaintext and the encryption
key. The _XOR_ algorithm is _not_ secure but on some special cases
(e.g., when the key has the same length of the plaintext, and the key
contains truly random bytes); however, it is certainly "good enough"
for this exercise.

_XOR_ is a _symmetric_ encryption algorithm, meaning that the same key
must be used for encrypting and decrypting a message. Indeed, the
exact same algorithm can be used to encrypt or decrypt a message, as
shown below.

The program contains a function `xorcrypt(in, out, n, key, keylen)`
that can be used to encrypt or decrypt a message with a given key. To
encrypt a message, then `in` points to the plaintext and `out` points
to a memory buffer that will contain the ciphertext. To decrypt a
message, then `in` points to the ciphertext and `out` points to a
memory buffer that will contain the plaintext.

The parameters are as follows:

- `in` points to the source message. This buffer does not need to be
  zero-terminated since it may contain arbitrary bytes;

- `out` points to a memory buffer of at least $n$ Bytes (the same
  length of the source message), that must be allocated by the caller.
  At the end, this buffer contains the source message that has been
  encrypted/decrypted with the encryption key;

- `n` is the length, in Bytes, of the source message;

- `key` points to the encryption/decryption key. The key is
  interpreted as a sequence of arbitrary bytes, and therefore does not
  need to be zero-terminated

- `keylen` is the length of the encryption/decryption key.

The _XOR_ algorithm will happily decrypt any message with any provided
key; of course, if the key is not correct, the decrypted message will
not make any sense. For this exercise the plaintext is a
zero-terminated ASCII string that can be printed with the `printf()`
function whose first ten characters are `"0123456789"`. This
information can be used to check whether you "guessed" the right
encryption key.

The encryption key that has been used in this program is a sequence of
8 ASCII numeric characters; therefore, the key is a string between
`"00000000"` and `"99999999"`. Write a program to brute-force the key
using MPI.

If $P$ MPI processes are used, then the key space is partitioned into
$P$ blocks, and the lower and upper bounds [_low_, _high_) of the
blocks to each process. Then, each process tries all keys within its
block.  This must be done carefully, since a process should
periodically check whether other processes have found the key. The
easiest way to achieve this is to explore the keys [_low_, _high_) in
chunks of size _BLKLEN_ (e.g., _BLKLEN_ = 1024). Every _BLKLEN_ keys,
each process checks whether the key has been found. If not, the next
chunk of _BLKLEN_ keys is tried.

To check whether the key has been found, it is possible to use a
_max-reduction_ operation. Each process sends the value -1 if it found
no key, or the key value otherwise. All processes compute the maximum
of the local keys. If the maximum is still -1, then no key has been
found.

Compile with:

        mpicc -std=c99 -Wall -Wpedantic mpi-brute-force.c -o mpi-brute-force

Run with:

        mpirun -n 4 ./mpi-brute-force

**Note**: the execution time of the parallel program might change
irregularly depending on the number $P$ of MPI processes. Why?

## Files

You can use the `wget` command to easily transfer the files on the lab
server:

        wget https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-brute-force.c

- [mpi-brute-force.c](mpi-brute-force.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

/* Decrypt `enc` of length `n` bytes into buffer `dec` using `key` of
   length `keylen`. The encrypted message, decrypted messages and key
   are treated as binary blobs; hence, they do not need to be
   zero-terminated.

   Do not modify this function. */
void xorcrypt(const char* in, char* out, int n, const char* key, int keylen)
{
    for (int i=0; i<n; i++) {
        out[i] = in[i] ^ key[i % keylen];
    }
}

/* Encrypt message `msg` using key `key` of length `keylen`. `mst`
   must be a zero-terminated ASCII string. Returns a pointer to a
   newly allocated block of length `(strlen(msg)+1)` containing the
   encrypted message. */
char *gen_encrypt( const char *msg, char *key, int keylen )
{
    const int n = strlen(msg)+1;
    char* out = malloc(n);
    int i;

    assert(out != NULL);
    xorcrypt(msg, out, n, key, keylen);
    printf("const char enc[] = {");
    for (i=0; i<n; i++) {
        if (i%8 == 0) {
            printf("\n");
        }
        printf("%d", out[i]);
        if ( i < n-1 ) {
            printf(", ");
        }
    }
    printf("\n};\n");
    return out;
}

int main( int argc, char *argv[] )
{
    const int KEY_LEN = 8;
    /* encrypted message */
    const char enc[] = {
        4, 1, 0, 1, 0, 1, 4, 1,
        12, 9, 115, 18, 71, 64, 64, 87,
        90, 87, 87, 18, 83, 85, 95, 83,
        26, 16, 102, 90, 81, 20, 93, 88,
        88, 73, 18, 69, 93, 90, 92, 95,
        90, 87, 18, 95, 91, 66, 87, 22,
        93, 67, 18, 92, 91, 64, 18, 66,
        91, 16, 66, 94, 85, 77, 28, 54
    };

    int my_rank, comm_sz;
    /* There is some redundant code that has been used by me to
       generate the encrypted message */
    const char *msg = "0123456789A strange game. The only winning move is not to play."; /* plaintext message */
    const int msglen = strlen(msg)+1; /* length of the encrypted message, including the trailing \0 */
#if 0
    char enc_key[] = "40224426"; /* encryption key */
#endif
    const int NUM_KEYS = 100000000;    /* total number of possible keys */
    const char check[] = "0123456789"; /* the decrypted message starts with this string */
    const int CHECK_LEN = strlen(check);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    char* buf = (char*)malloc(msglen);
    char key[KEY_LEN+1];
    int my_start = (NUM_KEYS*my_rank)/comm_sz;
    const int my_end = (NUM_KEYS*(my_rank+1))/comm_sz;
    int found = -1, local_found = -1;
    int k = my_start;

    const double tstart = MPI_Wtime();
    const int BLKLEN = 1024; /* processes synchronize every BLKLEN keys */
    const int N_ROUNDS = (NUM_KEYS + comm_sz*BLKLEN + 1)/(comm_sz*BLKLEN);
    int round = 0;

    /* We must be careful here: each process must perform the same
       maximum number of rounds. If it does not, then the Allreduce()
       might block because some processes exited the loop. */
    do {
        for (k = my_start; k < my_start + BLKLEN && k < my_end && local_found < 0; k++) {
	  sprintf(key, "%08u", (unsigned int)k);
            xorcrypt(enc, buf, msglen, key, 8);
            if ( 0 == memcmp(buf, check, CHECK_LEN)) {
                local_found = k;
            }
        }

        my_start += BLKLEN;

        MPI_Allreduce( &local_found,    /* sendbuf              */
                       &found,          /* recvbuf              */
                       1,               /* count                */
                       MPI_INT,         /* sent datatype        */
                       MPI_MAX,         /* operation            */
                       MPI_COMM_WORLD   /* communicator         */
                       );
        round++;
    } while (found < 0 && round < N_ROUNDS);

    const double elapsed = MPI_Wtime() - tstart;

    if ( 0 == my_rank ) {
        printf("Elapsed time: %f\n", elapsed);
        if (found < 0) {
            fprintf(stderr, "FATAL: key not found\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        } else {
            printf("Key found \"%d\"\n", found);
            sprintf(key, "%08d", found);
            xorcrypt(enc, buf, msglen, key, 8);
            printf("Decrypted message: %s\n", buf);
        }
    }

    free(buf);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
