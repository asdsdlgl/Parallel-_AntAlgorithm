 編譯方式 mpiicc -openmp -o hw6 hw6.cpp
 執行方式 mpiexec -n mpi個數 ./hw6 檔案名稱 城市數目 (如 :mpiexec -n 8 ./hw6 fri26_d.txt 26)
 基本上大約是15s左右
 注意 : 標頭黨 #include</usr/lib64/gcc/x86_64-suse-linux/4.5/include/omp.h>
 此為在老師的server上omp.h的位置 每個server可能不同!!
 利用在command line打find /usr -name omp.h 可找到omp位置 再更改head file就可以了