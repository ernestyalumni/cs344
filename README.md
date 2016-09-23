cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit. 
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

==================================================

Changes by me (Ernest Yeung)
==================================================

# Summary of changes

- *Added* (entirely *new*) Lesson 1 Code Snippets in [Lesson Code Snippets](https://github.com/ernestyalumni/cs344/tree/master/Lesson%20Code%20Snippets)


| codename          | directory                      | Keywords      | Description |
| ----------------- | :----------------------------- | :-----------: | ----------- | 
| `student_func00.cu` | `./Problem Sets/Problem Set 1/` | Problem Set 1 | My first attempt before I spent 2.5 months with CUDA C/C++ (about June 2016) |
| `student_func.cu` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my solution |
| `student_func00.cu` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my solution; has the "naive" gaussian blur method (i.e. from global memory) |
| `Makefile` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | changed Makefile to run on my Fedora Linux setup (mostly changed gcc to nvcc compiler, needed for `cuda_runtime.h` |
| `HW2` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | executable for Problem Set 2 for reference (of a working executable), using the "naive" gaussian blur method (no shared memory).  Results I obtained for running `./HW2 cinque_terre_small.jpg` was `Your code ran in: 1.595616 msecs` on a NVIDIA GTX GeForce 980 Ti, EVGA, for thread block size of 16x16, for 32x32, 1.514528 msecs; see the benchmarks below |
| `student_func_global.cu` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my final version implementing the "naive" gaussian blur method (i.e. from global memory) |



## more on Problem Set 2

My writeup is in [`CompPhys.pdf`](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf) of the [`LaTeXandpdfs` directory](https://github.com/ernestyalumni/CompPhys/tree/master/LaTeXandpdfs) of the [CompPhys github repository](https://github.com/ernestyalumni/CompPhys) - search for "On Problem Set 2".  

### on the "naive" gaussian blur method (i.e. from global memory)

This method was implemented in [`student_func_global.cu`](https://github.com/ernestyalumni/cs344/blob/master/Problem%20Sets/Problem%20Set%202/student_func_global.cu)

#### Benchmarks (global memory):

Doing `./HW2 cinque_terre_small.jpg`, using this "naive" gaussian blur method (i.e. from global memory only), on the *EVGA NVIDIA GeForce GTX 980 Ti*:

For a `dim3 blockSize(1,1)`, i.e. $(M_x,M_y)=(1,1)$

```  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg  
Your code ran in: 46.665215 msecs.  
PASS  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg  
Your code ran in: 46.587330 msecs.  
PASS  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg  
Your code ran in: 46.565727 msecs.  
PASS  
```

For a `dim3 blockSize(2,2)`

```  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 12.110336 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 12.115456 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 12.149376 msecs.
PASS
```

For a `dim3 blockSize(4,4)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 3.383424 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 3.057344 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 3.060672 msecs.
PASS
```

For a `dim3 blockSize(8,8)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.775840 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.781280 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.776864 msecs.
PASS
```

For a `dim3 blockSize(16,16)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.582560 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.569120 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.350688 msecs.
PASS
```

For a `dim3 blockSize(32,32)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.521856 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.530208 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.513664 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.514176 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.305984 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.297568 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.303168 msecs.
PASS
```

For block sizes greater than 32, we obtain an error.  So it appears that for this "naive" gaussian blur (i.e. only global memory), then setting `dim3 blockSize(16,16)` or `dim3 blockSize(32,32)` makes the code run the fastest.  