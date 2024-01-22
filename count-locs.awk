# This AWK script counts the number of lines of code for each variant
# of each program, and produces a table that is suitable for
# publication. The program parses the output of `cloc`.
#
# Usage:
#
# cloc --by-file --csv *.c *.cl *.cu | gawk -f count-locs.awk
#
# Written by Moreno Marzolla
# last updated on 2024-01-22
BEGIN {
    FS = ","
}

{
    prog_type = gensub("\\-.*$", "", "g", $2);
    prog_ext = gensub("^.*\\.", "", "g", $2);
    prog_name = gensub("^[^\\-]*\\-", "", "g", $2);
    prog_name = gensub("\\..+$", "", "g", prog_name);
    locs = $5;
    if (prog_type == "omp" || prog_type == "mpi" || prog_type == "opencl" || prog_type == "cuda") {
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
    printf("{\\bf Kernel} & {\\bf OpenMP} & {\\bf MPI} & {\\bf CUDA} & {\\bf OpenCL} \\\\\n");
    printf("\\midrule\n");
    # Change iteration order so that the following loop enumerate the
    # programs by name
    PROCINFO["sorted_in"] = "@ind_str_asc";
    for (prog in progs) {
        printf("%25s & %4d & %4d & %4d & %4d \\\\\n",
               prog,
               loc[prog, "omp"],
               loc[prog, "mpi"],
               loc[prog, "cuda"],
               loc[prog, "opencl"]);
    }
    printf("\\midrule\n");
    printf("%25s & %4d & %4d & %4d & %4d \\\\\n",
           "{\\bf Number of programs}",
           count["omp"],
           count["mpi"],
           count["cuda"],
           count["opencl"]);

    printf("%25s & %04d & %04d & %04d & %04d \\\\\n",
           "{\\bf Total LOCs}",
           total_locs["omp"],
           total_locs["mpi"],
           total_locs["cuda"],
           total_locs["opencl"]);
    printf("\\bottomrule\n");
}
