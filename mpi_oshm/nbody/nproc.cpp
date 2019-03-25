#include <stdio.h>
#include <sys/resource.h>

int main()
{
    struct rlimit rl;
    getrlimit(RLIMIT_NPROC, &rl);
    printf("%d\n", rl.rlim_cur);
}
