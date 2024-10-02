# This AWK script counts the number of lines of code for each variant
# of each program, and produces a table that is suitable for
# publication. The program parses the output of `cloc`.
#
# Usage:
#
# cloc --by-file --csv *.c *.cl *.cu | gawk -f count-locs.awk
#
# Written by Moreno Marzolla
# last updated on 2024-10-02

# If v > 0 the returns a string where v is formatted according
# to fmt; otherwise, returns the empty string.
function pretty_print(fmt, v)
{
    if (v > 0)
        return sprintf(fmt, v)
    else
        return "";
}

BEGIN {
    FS = ","
}

title["anneal"] = "ANNEAL cellular automaton";
patterns["anneal"] = "2D";

title["bbox"] = "Bounding Box of a set of rectangles";
patterns["bbox"] = "SG, RD";

title["bintree-walk"] = "Binary tree traversal";
patterns["bintree-walk"] = "task-level parallelism";

title["brute-force"] = "Brute-force password cracking";
patterns["brute-force"] = "EP"

title["bsearch"] = "Binary search";
patterns["bsearch"] = "DC, task-level parallelism";

title["c-ray"] = "Ray Tracing";
patterns["c-ray"] = "EP, SG";

title["cat-map"] = "Arnold's cat map";
patterns["cat-map"] = "EP";

title["cat-map-rectime"] = "Recurrence time of Arnold's cat map";
patterns["cat-map-rectime"] = "RD";

title["circles"] = "Area of the union of circles";
patterns["circles"] = "SG, RD";

title["coupled-oscillators"] = "Coupled harmonic hoscillators";
patterns["coupled-oscillators"] = "EP, RD";

title["denoise"] = "Image denoising using median filter";
patterns["denoise"] = "2D";

title["dot"] = "Dot product";
patterns["dot"] = "SG, RD";

title["edge-detect"] = "Sobel's edge detection filter";
patterns["edge-detect"] = "2D";

title["first-pos"] = "First occurrence of a key in unsorted array";
patterns["first-pos"] = "SG, RD";

title["floyd-warshall"] = "All-pair shortest paths (Floyd and Warshall)";
patterns["floyd-warshall"] = "EP";

title["inclusive-scan"] = "Inclusive scan";
patterns["inclusive-scan"] = "SC, P2P";

title["knapsack"] = "0-1 knapsack problem";
patterns["knapsack"] = "Irregular 1D";

title["letters"] = "Count character frequencies on text";
patterns["letters"] = "EP, RD";

title["levenshtein"] = "Levenshtein's edit distance of strings";
patterns["levenshtein"] = "2D, wavefront";

title["list-ranking"] = "List ranking";
patterns["list-ranking"] = "pointer jumping";

title["lookup"] = "All occurrences of a key in an unsorted array";
patterns["lookup"] = "EP, RD";

title["mandelbrot"] = "Mandelbrot set";
patterns["mandelbrot"] = "EP, LB";

title["mandelbrot-area"] = "Area of the Mandelbrot set";
patterns["mandelbrot-area"] = "EP, LB, RD";

title["map-levels"] = "Remap gray levels of image";
patterns["map-levels"] = "EP, SG";

title["matsum"] = "Matrix sum";
patterns["matsum"] = "EP, SG";

title["my-bcast"] = "Broadcast using P2P communications";
patterns["my-bcast"] = "P2P";

title["merge-sort"] = "Merge-Sort";
patterns["merge-sort"] = "DC, task-level parallelism";

title["nbody"] = "N-body simulation";
patterns["nbody"] = "EP, LP, RD";

title["odd-even"] = "Odd-even transposition sort";
patterns["odd-even"] = "SG, P2P";

title["pi"] = "Monte-Carlo estimation of the value of $\\pi$";
patterns["pi"] = "EP, RD";

title["reverse"] = "Array reversal";
patterns["reverse"] = "EP";

title["ring"] = "Ring communication";
patterns["ring"] = "P2P";

title["rotate-right"] = "Circular rotation of an array";
patterns["rotate-right"] = "SG, P2P";

title["rule30"] = "\"Rule 30\" cellular automaton";
patterns["rule20"] = "1D, P2P";

title["sat"] = "Brute-force SAT solver";
patterns["sat"] = "EP, RD";

title["schedule"] = "Dynamic loop scheduling";
patterns["schedule"] = "master-worker";

title["sieve"] = "Prime-counting function $\\pi(n)$";
patterns["sieve"] = "EP, RD";

{
    prog_type = gensub("\\-.*$", "", "g", $2);
    prog_ext = gensub("^.*\\.", "", "g", $2);
    prog_name = gensub("^[^\\-]*\\-", "", "g", $2);
    prog_name = gensub("\\..+$", "", "g", prog_name);
    locs = $5;
    if (title[prog_name] != "" &&
        (prog_type == "omp" || prog_type == "mpi" || prog_type == "opencl" || prog_type == "cuda")) {
        #printf("%s %s %d\n", prog_name, prog_type, $5);
        loc[prog_name,prog_type] += locs;
        total_locs[prog_type] += locs;
        progs[prog_name] = 1;
        if (prog_type != "opencl" || prog_ext == "c")
            count[prog_type] += 1;
    }
}

END {
    printf("\\toprule\n");
    printf("{\\bf Kernel} \t&\t {\\bf OpenMP} \t&\t {\\bf MPI} \t&\t {\\bf CUDA} \t&\t {\\bf OpenCL} \t&\t {\\bf Patterns} \\\\\n");
    printf("\\midrule\n");
    # Change iteration order so that the following loop enumerate the
    # programs by name
    PROCINFO["sorted_in"] = "@ind_str_asc";
    for (prog in progs) {
        printf("%35s \t&\t %4s \t&\t %4s \t&\t %4s \t&\t %4s \t&\t %s \\\\\n",
               title[prog],
               pretty_print("%4d", loc[prog, "omp"]),
               pretty_print("%4d", loc[prog, "mpi"]),
               pretty_print("%4d", loc[prog, "cuda"]),
               pretty_print("%4d", loc[prog, "opencl"]),
               patterns[prog]);
    }
    printf("\\midrule\n");
    printf("%35s \t&\t %4d \t&\t %4d \t&\t %4d \t&\t %4d \t&\t \\\\\n",
           "{\\bf Number of programs}",
           count["omp"],
           count["mpi"],
           count["cuda"],
           count["opencl"]);

    printf("%35s \t&\t %04d \t&\t %04d \t&\t %04d \t&\t %04d \t&\t \\\\\n",
           "{\\bf Total LOCs}",
           total_locs["omp"],
           total_locs["mpi"],
           total_locs["cuda"],
           total_locs["opencl"]);
    printf("\\bottomrule\n");
}
