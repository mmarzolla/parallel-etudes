/****************************************************************************
 *
 * mpi-brute-force.c - Brute-force password cracking
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-08

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
using OpenMP. The program tries every key until a valid message is
eventually found, i.e., a message that begins with `"0123456789"`. At
the end, the program must print the plaintext, which is a relevant
quote from an old film.

Use an `omp parallel` construct (not `parallel for`) whose body
contains code that assigns a suitable subset of the key space to each
OpenMP thread. Recall from the lectures that the `omp parallel`
construct applies to a _structured block_, that is, a block with a
single entry and a single exit point. Therefore, the thread who finds
the correct key can not exit from the block using `return`, `break` or
`goto` (the compiler should raise a compile-time error if any of these
constructs are used). However, we certainly do not want to wait until
all keys have been explored to terminate the program.  Therefore, you
should devise a clean mechanism to terminate the computation as soon
as the correct key has been found. You may not terminate the program
with `exit()`, `abort()` or similar.

Compile with:

        mpicc -std=c99 -Wall -Wpedantic mpi-brute-force.c -o mpi-brute-force

Run with:

        mpirun -n 4 ./mpi-brute-force

**Note**: the execution time of the parallel program might change
irregularly depending on the number $P$ of OpenMP threads. Why?

## Files

You can use the `wget` command to easily transfer the files on the lab
server:

        wget https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-brute-force.c

- [mpi-brute-force.c](mpi-brute-force.c)

***/

#include "hpc.h"
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
    const int BLKLEN = 1024;
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
    char enc_key[] = "40224426"; /* encryption key */
    const int n = 100000000;    /* total number of possible keys */
    int found = 0, local_found = 0;
    const char check[] = "0123456789"; /* the decrypted message starts with this string */
    const int CHECK_LEN = strlen(check);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( 0 == my_rank ) {
		char *tmp = gen_encrypt(msg, enc_key, KEY_LEN);

		free(tmp);
	}

    const double tstart = hpc_gettime();
     
	char* buf = (char*)malloc(msglen);
	char key[KEY_LEN+1];
	const int my_start = (n*my_rank)/comm_sz;
	const int my_end = (n*(my_rank+1))/comm_sz;

	/* Technically, there is a race condition updating the
	   variable `found`; however, the race condition is benign
	   because in the worst case it forces the other threads to
	   execute one more iteration than necessary. */
	for ( int i=my_start; i<my_end && !found && i<n; i++) {
				
		if ( i%BLKLEN == 0 ) {
			
			// MPI_Barrier(MPI_COMM_WORLD); // Could be usefull before the use of collectives on some supercomputers
			
			MPI_Allreduce( &local_found, 	/* sendbuf      	*/
						   &found,			/* recvbuf      	*/
						   1,				/* count 			*/
						   MPI_INT,			/* sent datatype 	*/
						   MPI_SUM,			/* operation 		*/
						   MPI_COMM_WORLD   /* communicator		*/
						 );
		}
		
		sprintf(key, "%08d", i);
		xorcrypt(enc, buf, msglen, key, 8);
		if ( 0 == memcmp(buf, check, CHECK_LEN)) {
			printf("Key found: %s, by rank: %d\n", key, my_rank);
			printf("Decrypted message: %s\n", buf);
			local_found = 1;
			
			const double elapsed = hpc_gettime() - tstart;
			printf("Rank: %d, elapsed time: %f\n", my_rank, elapsed);
			
			printf("Broadcasting that the key has been found\n");
		}
	}
	
	if ( found > 1 ) {
		fprintf(stderr, "More than one key found - ERROR\n");
        return EXIT_FAILURE;
	}
	
	if ( !local_found && found ) {
		printf("My rank: %d, another process has found the key - EXITING\n", my_rank);
	}
	
	free(buf);
    
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}