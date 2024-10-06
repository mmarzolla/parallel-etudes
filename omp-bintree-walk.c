/****************************************************************************
 *
 * omp-bintree-walk.c - Parallel Binary Search Tree traversal with OpenMP tasks
 *
 * Copyright (C) 2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% HPC - Parallel Binary Search Tree traversal with OpenMP tasks
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2023-07-08

The program [omp-bintree-walk.c](omp-bintree-walk.c) creates a random
Binary Search Tree (BST) $T$ and performs a post-order visit of $T$
using a recursive algorithm.

Your goal is to parallelize the code using OpenMP tasks; the program
should create as many tasks as nodes in $T$, and each task should be
assigned one node to visit. The actual order in which $T$ is explored
is not important.

## Files

- [omp-bintree-walk.c](omp-bintree-walk.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <omp.h>

typedef struct bintree_node_t {
    int val;
    struct bintree_node_t *left, *right;
} bintree_node_t;

/* Inserts the value `val` into the subtree rooted at `root`. Returns
   the (possibly new) root of the subtree. `root` can be `NULL` if
   inserting into an empty subtree.  */
bintree_node_t *bt_insert(int val, bintree_node_t *root)
{
    if (root == NULL) {
        root = (bintree_node_t*)malloc(sizeof(bintree_node_t));
        assert(root != NULL);
        root->val = val;
        root->left = root->right = NULL;
    } else {
        if (val < root->val)
            root->left = bt_insert(val, root->left);
        else
            root->right = bt_insert(val, root->right);
    }
    return root;
}

/* Post-order visit of the subtree rooted at `root` */
void bt_visit(const bintree_node_t *root)
{
    if (root == NULL)
        return;
    else {
#ifndef SERIAL
#pragma omp task firstprivate(root)
#endif
        bt_visit(root->left);
#ifndef SERIAL
#pragma omp task firstprivate(root)
#endif
        bt_visit(root->right);
        printf("Thread %d visits %d\n", omp_get_thread_num(), root->val);
        sleep(1); /* inserts a delay */
    }
}

void bt_destroy(bintree_node_t *root)
{
    if (root != NULL) {
        bt_destroy(root->left);
        bt_destroy(root->right);
        free(root);
    }
}

/* Returns a random integer in [a, b] */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

void random_shuffle(int *v, int n)
{
    int i;
    for (i=0; i<n-1; i++) {
        const int j = randab(i, n-1);
        const int tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
    }
}

/* Create a binary search tree with `n` nodes; the tree contains a
   random permutation of the integers [0, .. n-1] */
bintree_node_t *bt_create(int n)
{
    int i;
    int *v = (int*)malloc(n * sizeof(*v));
    bintree_node_t *root = NULL;

    assert(v != NULL);

    for (i=0; i<n; i++)
        v[i] = i;

    random_shuffle(v, n);

    /* build the tree */
    for (i=0; i<n; i++) {
        root = bt_insert(v[i], root);
    }
    free(v);

    return root;
}

int main( int argc, const char *argv[] )
{
    int n = 1000;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [number of nodes]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2)
        n = atoi(argv[1]);

    bintree_node_t *root = bt_create(n);

#ifdef SERIAL
    bt_visit(root);
#else
#pragma omp parallel
    {
#pragma omp master
        bt_visit(root);
    }
#endif
    bt_destroy(root);

    return EXIT_SUCCESS;
}
